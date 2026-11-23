"""AI 阶段后台任务: 校对 / 会议理解 / 材料生成。

独立队列 + 并发上限(默认 2 个任务同时跑; 校对内部按批并行, 全局限流由
pipeline._llm_gate 控制)。任务状态/进度/部分结果都落 SQLite:
- 校对每批完成即回填, 失败或取消不丢已完成的批次;
- 理解/成文为整体结果, 失败不写半成品;
- 服务重启: running → failed(可重新发起), queued 重新入队。
"""
import asyncio
import json

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
        else:
            raise ValueError(f"未知任务类型: {job['kind']}")

    @staticmethod
    def _progress_cb(store, jid: str, label: str):
        def cb(done: int, total: int) -> None:
            total = max(1, total)
            store.update_job(jid, progress=round(done / total, 3),
                             detail=f"{label} {done}/{total}")
        return cb

    async def _proofread(self, jid: str, tid: str, params: dict) -> None:
        task = await asyncio.to_thread(self.store.get_task, tid)
        if not task or task["deleted"]:
            raise pipeline.LLMError("任务不存在")
        if task["status"] != "done":
            raise pipeline.LLMError("转写未完成")
        segs = await asyncio.to_thread(self.store.get_segments, tid)
        if not segs:
            raise pipeline.LLMError("无转写片段")
        counter = {"n": 0}
        force_human = bool(params.get("force_human"))

        def on_result(got: dict) -> None:
            counter["n"] += self.store.apply_corrections(
                tid, got, "llm", skip_human=not force_human)

        self.store.update_job(jid, detail="")
        _, stats = await asyncio.to_thread(
            pipeline.proofread_segments, segs, task["hotword"] or "",
            progress=self._progress_cb(self.store, jid, "已校对"),
            cancel=lambda: self._cancel_requested(jid),
            only_missing=bool(params.get("only_missing")),
            force_human=force_human, on_result=on_result)
        if not stats.get("cancelled") and counter["n"] == 0 and stats.get("failed_batches"):
            raise pipeline.LLMError("全部批次失败: " + "; ".join(stats["errors"][:2]))
        remaining = await asyncio.to_thread(self.store.count_missing_corrections, tid)
        result = {
            "corrected_count": counter["n"],
            "remaining_missing": remaining,
            "stats": stats,
        }
        blob = json.dumps(result, ensure_ascii=False)
        if stats.get("cancelled"):
            self.store.update_job(jid, status="cancelled", detail="已保存部分结果",
                                  result=blob)
            return
        detail = f"完成: 校对 {counter['n']} 段"
        if remaining:
            detail += f", {remaining} 段未覆盖"
        elif stats.get("failed_batches"):
            detail += f", {stats['failed_batches']} 批失败"
        self.store.update_job(jid, status="done", progress=1.0, detail=detail, result=blob)

    async def _understand(self, jid: str, tid: str) -> None:
        segs = await asyncio.to_thread(self.store.get_segments, tid)
        if not segs:
            raise pipeline.LLMError("无转写片段")
        speakers = {s["speaker_id"]: s["display_name"]
                    for s in self.store.get_speakers(tid)}
        self.store.update_job(jid, detail="")
        data = await asyncio.to_thread(
            pipeline.understand_meeting, segs, speakers,
            progress=self._progress_cb(self.store, jid, "已分析"),
            cancel=lambda: self._cancel_requested(jid))
        content = json.dumps(data, ensure_ascii=False)
        doc_id = self.store.put_document(
            tid, "understanding", content,
            {"_input_hash": pipeline.transcript_hash(segs, speakers)})
        counts = {k: len(data.get(k) or []) for k in
                  ("discussion", "viewpoints", "disagreements", "decisions",
                   "suggestions", "tasks", "unconfirmed")}
        result = {"document_id": doc_id, "counts": counts,
                  "failed_chunks": (data.get("_stats") or {}).get("failed_chunks", 0)}
        self.store.update_job(jid, status="done", progress=1.0, detail="完成",
                              result=json.dumps(result, ensure_ascii=False))

    async def _document(self, jid: str, tid: str, params: dict) -> None:
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
        self.store.update_job(jid, detail="", progress=0.1)
        doc = await asyncio.to_thread(
            pipeline.generate_document, kind, understanding, transcript, options,
            cancel=lambda: self._cancel_requested(jid))
        doc_id = self.store.put_document(
            tid, kind, doc,
            {**options, "_input_hash": pipeline.transcript_hash(segs, speakers)})
        result = {"document_id": doc_id, "kind": kind, "content": doc,
                  "chars": len(doc)}
        self.store.update_job(jid, status="done", progress=1.0,
                              detail=f"完成: {len(doc)} 字",
                              result=json.dumps(result, ensure_ascii=False))
