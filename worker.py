"""任务队列: asyncio 单消费者 + 分级信号量 + 重启恢复 + 协作式取消。

约束: ASR 串行(同一张 GPU); 解码并发 2; LLM 并发 2。
取消仅对未开始/解码阶段生效(ASR 是单次阻塞调用, 处理中任务不可中断, UI 需明示)。
"""
import asyncio
import shutil
from pathlib import Path

import config
import pipeline
from store import Store


class Worker:
    def __init__(self, store: Store):
        self.store = store
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self.decode_sem = asyncio.Semaphore(config.DECODE_CONCURRENCY)
        self.asr_sem = asyncio.Semaphore(config.ASR_CONCURRENCY)
        self.llm_sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
        self._consumer: asyncio.Task | None = None

    def start(self) -> None:
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.get_running_loop().create_task(self._loop())

    async def enqueue(self, tid: str) -> None:
        await self.q.put(tid)

    async def recover(self) -> dict:
        info = await asyncio.to_thread(self.store.recover_on_startup)
        for tid in await asyncio.to_thread(self.store.pending_queued):
            await self.enqueue(tid)
        return info

    async def _loop(self) -> None:
        while True:
            tid = await self.q.get()
            try:
                await self._process(tid)
            except Exception as e:  # noqa: BLE001 worker 不允许死
                self.store.update_task(tid, status="failed",
                                       error=f"{type(e).__name__}: {e}", stage="done")
            finally:
                self.q.task_done()

    async def _process(self, tid: str) -> None:
        task = await asyncio.to_thread(self.store.get_task, tid)
        if not task or task["status"] != "queued" or task["deleted"]:
            return
        if task["cancel_requested"]:
            self.store.update_task(tid, status="cancelled", stage="done")
            return
        self.store.update_task(tid, status="processing")
        workdir = Path(config.TMP_DIR) / f"job-{tid}"
        workdir.mkdir(parents=True, exist_ok=True)
        wav = workdir / f"{tid}.wav"
        try:
            # 源: 优先已解码 wav(重试免二次解码), 否则回退上传原始文件
            ap = task.get("audio_path") or ""
            if ap.endswith(".wav") and Path(ap).exists():
                duration = task["duration_sec"]
                shutil.copy(ap, str(wav))
            else:
                src = Path(config.UPLOAD_DIR) / f"{tid}.bin"
                if not src.exists():
                    raise ValueError("源音频文件已清理, 请重新上传")
                self.store.update_task(tid, stage="decoding")
                async with self.decode_sem:
                    duration = await asyncio.to_thread(
                        pipeline.decode_audio, str(src), str(wav))
            if (await asyncio.to_thread(self.store.get_task, tid))["cancel_requested"]:
                self.store.update_task(tid, status="cancelled", stage="done",
                                       duration_sec=duration)
                return
            self.store.update_task(tid, stage="asr", duration_sec=duration)
            async with self.asr_sem:
                segments = await asyncio.to_thread(
                    pipeline.transcribe_wav, str(wav), task["hotword"] or "")
            audio_path = Path(config.AUDIO_DIR) / f"{tid}.wav"
            shutil.move(str(wav), str(audio_path))
            await asyncio.to_thread(self.store.replace_segments, tid, segments)
            for spk in sorted({s["speaker_id"] for s in segments if s["speaker_id"] is not None}):
                self.store.upsert_speaker(tid, spk, f"说话人{spk + 1}")
            self.store.update_task(tid, status="done", stage="done",
                                   audio_path=str(audio_path))
        except ValueError as e:  # 解码类明确错误
            self.store.update_task(tid, status="failed", error=str(e), stage="done")
        except Exception as e:  # noqa: BLE001
            self.store.update_task(tid, status="failed",
                                   error=f"{type(e).__name__}: {e}", stage="done")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
