"""AI 阶段后台任务: 校对 / 会议理解 / 材料生成 / 一键流水线。

独立队列 + 并发上限(默认 2 个任务同时跑; 校对内部按批并行, 全局限流由
pipeline._llm_gate 控制)。任务状态/进度/部分结果都落 SQLite:
- 校对每批完成即回填, 失败或取消不丢已完成的批次;
- 理解/成文为整体结果, 失败不写半成品;
- 一键流水线 = 校对(默认只补未校段) → 理解 → 材料, 任一步失败保留已完成成果;
- 服务重启: running → failed(可重新发起), queued 重新入队。
"""
import asyncio
import json
import re

import config
import pipeline


class AIWorker:
    def __init__(self, store):
        self.store = store
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self._sem = asyncio.Semaphore(max(1, config.AI_JOB_CONCURRENCY))
        self._consumer: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.get_running_loop().create_task(self._loop())

    async def enqueue(self, job_id: str) -> None:
        await self.q.put(job_id)

    async def recover(self) -> dict:
        info = await asyncio.to_thread(self.store.recover_jobs)
        for jid in await asyncio.to_thread(self.store.pending_queued_jobs):
            await self.enqueue(jid)
        return info

    async def _loop(self) -> None:
        while True:
            jid = await self.q.get()
            task = asyncio.get_running_loop().create_task(self._run(jid))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            self.q.task_done()

    def _cancel_requested(self, jid: str) -> bool:
        job = self.store.get_job(jid)
        return bool(job and job["cancel_requested"])

    async def _run(self, jid: str) -> None:
        async with self._sem:
            job = await asyncio.to_thread(self.store.get_job, jid)
            if not job or job["status"] != "queued":
                return
            if job["cancel_requested"]:
                self.store.update_job(jid, status="cancelled", detail="")
                return
            self.store.update_job(jid, status="running", progress=0.0, detail="")
            try:
                await self._dispatch(job)
            except pipeline.Cancelled:
                self.store.update_job(jid, status="cancelled", detail="")
            except Exception as e:  # noqa: BLE001 AI 任务失败必须落状态
                self.store.update_job(jid, status="failed", detail="失败",
                                      error=f"{type(e).__name__}: {e}"[:500])

    async def _dispatch(self, job: dict) -> None:
        jid = job["id"]
        tid = job["task_id"]
        params = json.loads(job["params"] or "{}")
        if job["kind"] == "proofread":
            await self._proofread(jid, tid, params)
        elif job["kind"] == "understand":
            await self._understand(jid, tid)
        elif job["kind"] == "document":
            await self._document(jid, tid, params)
        elif job["kind"] == "pipeline":
            await self._pipeline(jid, tid, params)
        else:
            raise ValueError(f"未知任务类型: {job['kind']}")

    @staticmethod
    def _progress_cb(store, jid: str, label: str):
        def cb(done: int, total: int) -> None:
            total = max(1, total)
            store.update_job(jid, progress=round(done / total, 3),
                             detail=f"{label} {done}/{total}")
        return cb

    # ---------------- 可复用的阶段执行体 ----------------

    async def _do_proofread(self, tid: str, params: dict, *, progress=None,
                            cancel=None) -> dict:
        task = await asyncio.to_thread(self.store.get_task, tid)
        if not task or task["deleted"]:
            raise pipeline.LLMError("任务不存在")
        if task["status"] != "done":
            raise pipeline.LLMError("转写未完成")
        segs = await asyncio.to_thread(self.store.get_segments, tid)
        if not segs:
            raise pipeline.LLMError("无转写片段")
        glossary = await asyncio.to_thread(self.store.glossary)
        rights = await asyncio.to_thread(self.store.glossary_rights)
        hotword = " ".join(dict.fromkeys(
            ([task["hotword"].strip()] if task["hotword"] else []) + rights))
        counter = {"n": 0}
        force_human = bool(params.get("force_human"))

        def on_result(got: dict) -> None:
            counter["n"] += self.store.apply_corrections(
                tid, got, "llm", skip_human=not force_human)

        _, stats = await asyncio.to_thread(
            pipeline.proofread_segments, segs, hotword,
            progress=progress, cancel=cancel,
            only_missing=bool(params.get("only_missing")),
            force_human=force_human, on_result=on_result, glossary=glossary)
        if not stats.get("cancelled") and counter["n"] == 0 and stats.get("failed_batches"):
            raise pipeline.LLMError("全部批次失败: " + "; ".join(stats["errors"][:2]))
        remaining = await asyncio.to_thread(self.store.count_missing_corrections, tid)
        return {"corrected_count": counter["n"], "remaining_missing": remaining,
                "stats": stats}

    async def _do_understand(self, tid: str, *, progress=None, cancel=None) -> dict:
        segs = await asyncio.to_thread(self.store.get_segments, tid)
        if not segs:
            raise pipeline.LLMError("无转写片段")
        speakers = {s["speaker_id"]: s["display_name"]
                    for s in self.store.get_speakers(tid)}
        data = await asyncio.to_thread(
            pipeline.understand_meeting, segs, speakers,
            progress=progress, cancel=cancel)
        content = json.dumps(data, ensure_ascii=False)
        doc_id = self.store.put_document(
            tid, "understanding", content,
            {"_input_hash": pipeline.transcript_hash(segs, speakers)})
        # 首次理解时用会议主题自动命名任务, 列表不再全是 meeting.wav
        task = self.store.get_task(tid)
        topic = (data.get("topic") or "").strip()
        if task and not (task.get("title") or "").strip() \
                and topic and topic != "未识别会议主题":
            self.store.update_task(tid, title=topic[:80])
        counts = {k: len(data.get(k) or []) for k in
                  ("discussion", "viewpoints", "disagreements", "decisions",
                   "suggestions", "tasks", "unconfirmed")}
        return {"document_id": doc_id, "counts": counts,
                "failed_chunks": (data.get("_stats") or {}).get("failed_chunks", 0),
                "topic": topic}

    async def _do_document(self, tid: str, params: dict, *, cancel=None) -> dict:
        kind = params.get("kind") or ""
        if kind not in pipeline._DOC_KINDS:
            raise pipeline.LLMError(f"kind 必须是 {list(pipeline._DOC_KINDS)}")
        docs = await asyncio.to_thread(self.store.get_documents, tid, "understanding")
        if not docs:
            raise pipeline.LLMError("请先生成会议理解(understand)")
        understanding = json.loads(docs[0]["content"])
        segs = await asyncio.to_thread(self.store.get_segments, tid)
        speakers = {s["speaker_id"]: s["display_name"]
                    for s in self.store.get_speakers(tid)}
        transcript = pipeline.format_transcript(segs, speakers)
        options = params.get("options") or {}
        doc = await asyncio.to_thread(
            pipeline.generate_document, kind, understanding, transcript, options,
            cancel=cancel)
        valid = {s["idx"] for s in segs}
        doc = pipeline.clean_citations(doc, valid)
        citations = len(re.findall(r"\[#\d+\]", doc))
        doc_id = self.store.put_document(
            tid, kind, doc,
            {**options, "_input_hash": pipeline.transcript_hash(segs, speakers),
             "_citations": citations})
        return {"document_id": doc_id, "kind": kind, "content": doc,
                "chars": len(doc), "citations": citations}

    # ---------------- 单阶段 job ----------------

    async def _proofread(self, jid: str, tid: str, params: dict) -> None:
        result = await self._do_proofread(
            tid, params,
            progress=self._progress_cb(self.store, jid, "已校对"),
            cancel=lambda: self._cancel_requested(jid))
        blob = json.dumps(result, ensure_ascii=False)
        if result["stats"].get("cancelled"):
            self.store.update_job(jid, status="cancelled", detail="已保存部分结果",
                                  result=blob)
            return
        detail = f"完成: 校对 {result['corrected_count']} 段"
        remaining = result["remaining_missing"]
        if remaining:
            detail += f", {remaining} 段未覆盖"
        elif result["stats"].get("failed_batches"):
            detail += f", {result['stats']['failed_batches']} 批失败"
        self.store.update_job(jid, status="done", progress=1.0, detail=detail, result=blob)

    async def _understand(self, jid: str, tid: str) -> None:
        self.store.update_job(jid, detail="分析中…")
        result = await self._do_understand(
            tid, progress=self._progress_cb(self.store, jid, "已分析"),
            cancel=lambda: self._cancel_requested(jid))
        self.store.update_job(jid, status="done", progress=1.0, detail="完成",
                              result=json.dumps(result, ensure_ascii=False))

    async def _document(self, jid: str, tid: str, params: dict) -> None:
        self.store.update_job(jid, detail="生成中…", progress=0.1)
        result = await self._do_document(tid, params,
                                         cancel=lambda: self._cancel_requested(jid))
        self.store.update_job(jid, status="done", progress=1.0,
                              detail=f"完成: {result['chars']} 字",
                              result=json.dumps(result, ensure_ascii=False))

    # ---------------- 一键流水线: 校对 → 理解 → 材料 ----------------

    async def _pipeline(self, jid: str, tid: str, params: dict) -> None:
        kind = params.get("kind") or "minutes"
        if kind not in pipeline._DOC_KINDS:
            raise pipeline.LLMError(f"kind 必须是 {list(pipeline._DOC_KINDS)}")
        options = params.get("options") or {}

        def set_progress(frac: float, detail: str) -> None:
            self.store.update_job(jid, progress=round(min(0.99, frac), 3), detail=detail)

        set_progress(0.01, "1/3 校对中…")
        p = await self._do_proofread(
            tid, {"only_missing": bool(params.get("only_missing", True))},
            progress=lambda d, t: set_progress(0.45 * d / max(1, t), f"1/3 校对 {d}/{t}"),
            cancel=lambda: self._cancel_requested(jid))
        if p["stats"].get("cancelled"):
            raise pipeline.Cancelled("已取消")
        set_progress(0.46, "2/3 理解中…")
        u = await self._do_understand(
            tid,
            progress=lambda d, t: set_progress(0.46 + 0.24 * d / max(1, t),
                                               f"2/3 理解 {d}/{t}"),
            cancel=lambda: self._cancel_requested(jid))
        set_progress(0.72, "3/3 生成材料…")
        d = await self._do_document(tid, {"kind": kind, "options": options},
                                    cancel=lambda: self._cancel_requested(jid))
        result = {"proofread": p, "understanding": u, "document": d}
        detail = (f"完成: 校对 {p['corrected_count']} 段, "
                  f"理解 {sum(u['counts'].values())} 条, {d['chars']} 字")
        self.store.update_job(jid, status="done", progress=1.0, detail=detail,
                              result=json.dumps(result, ensure_ascii=False))
