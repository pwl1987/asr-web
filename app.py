"""ASR-Web 会议工作台 — FastAPI 入口。

鉴权: 除 /login 与 /api/ping 外全部需要 Token(cookie `asr_token` / Bearer / ?token=)。
Token 来源: 环境变量 ASR_WEB_TOKEN, 未设置则自动生成并持久化到 data/.auth-token(0600)。
AI 阶段(校对/理解/成文)为后台任务: POST 返回 job_id, GET /api/jobs/{id} 轮询进度/结果。
"""
import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

import config
import pipeline
from docx_export import markdown_to_docx
from ai_worker import AIWorker
from store import Store
from worker import Worker

# ---------------------------------------------------------------- token

def _resolve_token() -> str:
    if config.ASR_WEB_TOKEN:
        return config.ASR_WEB_TOKEN
    if config.TOKEN_FILE.exists():
        t = config.TOKEN_FILE.read_text().strip()
        if t:
            return t
    t = secrets.token_urlsafe(24)
    config.ensure_dirs()
    config.TOKEN_FILE.write_text(t)
    os.chmod(config.TOKEN_FILE, 0o600)
    return t


class _TokenRedactor(logging.Filter):
    """uvicorn 访问日志里把 ?token=... 脱敏, 避免凭据进 journal。"""

    _PATTERN = re.compile(r"(token=)[^&\s\"']+")

    def filter(self, record):
        args = record.args
        if isinstance(args, (tuple, list)):
            try:
                record.args = type(args)(
                    self._PATTERN.sub(r"\1***", a) if isinstance(a, str) else a
                    for a in args)
            except Exception:  # noqa: BLE001 日志脱敏不允许影响业务
                pass
        return True


for _logger in ("uvicorn.access", "uvicorn.error"):
    logging.getLogger(_logger).addFilter(_TokenRedactor())


config.ensure_dirs()
AUTH_TOKEN = _resolve_token()
COOKIE = "asr_token"

store = Store(config.DB_PATH)
worker = Worker(store)
ai_worker = AIWorker(store)

LOGIN_HTML = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录 · 语音转写工作台</title><style>
body{margin:0;background:#0e1013;color:#e8eaf0;font:14px/1.6 -apple-system,"PingFang SC","Noto Sans SC",sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh}
.c{background:#15181d;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:28px;width:320px}
h1{font-size:16px;margin:0 0 16px;font-weight:600}
input{width:100%;box-sizing:border-box;background:#1a1e25;border:1px solid rgba(255,255,255,.12);
color:#e8eaf0;border-radius:9px;padding:10px 12px;font-size:14px;outline:none;margin-bottom:12px}
button{width:100%;background:linear-gradient(135deg,#6a8dff,#9b6aff);border:none;color:#fff;
border-radius:9px;padding:10px;font-size:14px;font-weight:600;cursor:pointer}
.err{color:#ef6a6a;font-size:12px;margin-top:10px;display:none}
</style></head><body><div class="c"><h1>语音转写工作台</h1>
<form method="post" action="/login">
<input type="password" name="token" placeholder="访问令牌" autofocus>
<button type="submit">进入</button></form>
<div class="err" id="e">令牌错误</div></div></body></html>"""


@asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, pipeline.load_asr_model)  # 后台加载
    worker.start()
    ai_worker.start()
    info = await worker.recover()
    if info.get("interrupted"):
        print(f"[asr-web] startup recovery: interrupted tasks -> {info['interrupted']}")
    jinfo = await ai_worker.recover()
    if jinfo.get("interrupted"):
        print(f"[asr-web] startup recovery: interrupted AI jobs -> {jinfo['interrupted']}")
    yield


app = FastAPI(title="asr-web meeting workbench", lifespan=_lifespan)

# ---------------------------------------------------------------- 鉴权中间件

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in ("/api/ping", "/login"):
        return await call_next(request)
    if path == "/favicon.ico":
        return Response(status_code=204)
    token = request.cookies.get(COOKIE) or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.query_params.get("token", "")
    if not hmac.compare_digest(token, AUTH_TOKEN):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "未授权"}, status_code=401)
        return RedirectResponse("/login", status_code=302)
    return await call_next(request)


@app.get("/login")
async def login_page():
    return HTMLResponse(LOGIN_HTML)


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    token = str(form.get("token") or "")
    if not hmac.compare_digest(token, AUTH_TOKEN):
        await asyncio.sleep(1.0)  # 简单防爆破
        return HTMLResponse(LOGIN_HTML.replace(
            'class="err" id="e"', 'class="err" id="e" style="display:block"'))
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, AUTH_TOKEN, max_age=30 * 24 * 3600,
                    httponly=True, samesite="lax")
    return resp


# ---------------------------------------------------------------- 健康/探活

@app.get("/api/ping")
async def ping():
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"model_loaded": pipeline.asr_ready(), "model_error": pipeline.asr_error(),
            "auth": True, "llm_model": config.LLM_MODEL,
            "limits": {"max_upload_mb": config.MAX_UPLOAD_MB,
                       "max_duration_min": config.MAX_DURATION_MIN,
                       "asr_concurrency": config.ASR_CONCURRENCY,
                       "ai_job_concurrency": config.AI_JOB_CONCURRENCY}}


# ---------------------------------------------------------------- 任务: 创建/列表/详情/操作

@app.post("/api/tasks")
async def create_task(request: Request, file: UploadFile = File(...), hotword: str = Form("")):
    filename = os.path.basename(file.filename or "recording.bin") or "recording.bin"
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    declared = 0
    raw_len = request.headers.get("content-length")
    if raw_len is not None:
        try:
            declared = int(raw_len)
        except ValueError:
            raise HTTPException(400, "Content-Length 非法")
        if declared < 0:
            raise HTTPException(400, "Content-Length 非法")
        if declared > max_bytes + 64 * 1024:
            raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB}MB 上限")
    tid = store.create_task(filename, declared, (hotword or "").strip())
    upload_path = Path(config.UPLOAD_DIR) / f"{tid}.bin"
    size = 0
    try:
        with open(upload_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB}MB 上限")
                f.write(chunk)
    except Exception:
        upload_path.unlink(missing_ok=True)
        store.purge_task(tid)
        raise
    if size == 0:
        upload_path.unlink(missing_ok=True)
        store.purge_task(tid)
        raise HTTPException(400, "空文件")
    store.update_task(tid, size_bytes=size)
    # 流式后去重(按实际字节数; Content-Length 含 multipart 开销, 不能直接比)
    existing = await asyncio.to_thread(store.find_active_by_file, filename, size)
    if existing and existing != tid:
        upload_path.unlink(missing_ok=True)
        store.purge_task(tid)
        return {"task_id": existing, "deduplicated": True}
    await worker.enqueue(tid)
    return {"task_id": tid, "size_bytes": size}


@app.get("/api/tasks")
async def list_tasks(q: str = ""):
    return store.list_tasks(q=q)


@app.get("/api/tasks/{tid}")
async def get_task(tid: str):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    return t


@app.patch("/api/tasks/{tid}")
async def patch_task(tid: str, payload: dict):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    if "title" not in payload:
        raise HTTPException(400, "只支持修改 title")
    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "标题不能为空")
    if len(title) > 80:
        raise HTTPException(400, "标题最多 80 字")
    store.update_task(tid, title=title)
    return {"title": title}


@app.post("/api/tasks/{tid}/cancel")
async def cancel_task(tid: str):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    if t["status"] == "queued":
        store.update_task(tid, status="cancelled", stage="done")
        return {"status": "cancelled"}
    if t["status"] == "processing":
        store.update_task(tid, cancel_requested=1)
        return {"status": "cancelling", "note": "解码阶段结束后生效; ASR 处理中不可中断"}
    raise HTTPException(409, f"状态 {t['status']} 不可取消")


@app.post("/api/tasks/{tid}/retry")
async def retry_task(tid: str):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    if t["status"] not in ("failed", "cancelled"):
        raise HTTPException(409, f"状态 {t['status']} 不可重试")
    has_wav = t["audio_path"] and t["audio_path"].endswith(".wav") \
        and Path(t["audio_path"]).exists()
    has_src = (Path(config.UPLOAD_DIR) / f"{tid}.bin").exists()
    if not has_wav and not has_src:
        raise HTTPException(400, "音频文件已清理, 请重新上传")
    store.update_task(tid, status="queued", stage="uploaded", error=None, cancel_requested=0)
    await worker.enqueue(tid)
    return {"status": "queued"}


@app.post("/api/tasks/{tid}/delete")
async def delete_task(tid: str):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    if t["status"] == "processing":
        raise HTTPException(409, "处理中的任务不可删除, 请先等待完成或取消")
    if store.has_active_job(tid):
        raise HTTPException(409, "有 AI 任务进行中, 请先取消后再删除")
    info = await asyncio.to_thread(store.purge_task, tid)
    removed = []
    for p in (info.get("audio_path"), str(Path(config.UPLOAD_DIR) / f"{tid}.bin")):
        if p and Path(p).exists():
            Path(p).unlink(missing_ok=True)
            removed.append(p)
    return {"deleted": True, "removed_files": removed}


# ---------------------------------------------------------------- 片段/说话人/编辑

@app.get("/api/tasks/{tid}/segments")
async def get_segments(tid: str):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    return {"task": t, "segments": store.get_segments(tid),
            "speakers": store.get_speakers(tid), "edits": store.get_edits(tid)}


@app.patch("/api/segments/{seg_id}")
async def patch_segment(seg_id: int, payload: dict):
    tid = payload.get("task_id") or ""
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    segs = store.get_segments(tid)
    seg = next((s for s in segs if s["id"] == seg_id), None)
    if not seg:
        raise HTTPException(404, "片段不存在")
    changed = {}
    if "text" in payload:
        new = str(payload["text"] or "")
        store.update_segment(seg_id, tid, "text", new, "human")
        changed["text"] = new
    if "speaker_id" in payload:
        sid = payload["speaker_id"]
        if isinstance(sid, bool) or (sid is not None and
                                     (not isinstance(sid, int) or sid < 0 or sid > 999)):
            raise HTTPException(400, "speaker_id 非法")
        store.update_segment(seg_id, tid, "speaker_id", sid, "human")
        if sid is not None:
            names = {s["speaker_id"] for s in store.get_speakers(tid)}
            if sid not in names:
                store.upsert_speaker(tid, sid, f"说话人{sid + 1}")
        changed["speaker_id"] = sid
    if not changed:
        raise HTTPException(400, "无可更新字段")
    return {"updated": changed}


@app.post("/api/tasks/{tid}/speakers/{sid}/rename")
async def rename_speaker(tid: str, sid: int, payload: dict):
    name = (payload.get("display_name") or "").strip()
    if not name:
        raise HTTPException(400, "display_name 为空")
    store.upsert_speaker(tid, sid, name)
    return {"speaker_id": sid, "display_name": name}


@app.post("/api/tasks/{tid}/speakers/merge")
async def merge_speakers(tid: str, payload: dict):
    from_ids = payload.get("from") or []
    to_id = payload.get("to")
    if not from_ids or to_id is None or to_id in from_ids:
        raise HTTPException(400, "参数非法: 需 from 数组与目标 to")
    n = store.merge_speakers(tid, from_ids, to_id)
    return {"moved_segments": n}


@app.post("/api/tasks/{tid}/speakers/split")
async def split_speaker(tid: str, payload: dict):
    seg_ids = payload.get("segment_ids") or []
    if not seg_ids:
        raise HTTPException(400, "segment_ids 为空")
    segs = store.get_segments(tid)
    max_sid = max([s["speaker_id"] for s in segs if s["speaker_id"] is not None] or [-1])
    new_sid = max_sid + 1
    n = store.reassign_speakers(tid, seg_ids, None, new_speaker_id=new_sid)
    return {"new_speaker_id": new_sid, "moved_segments": n}


@app.post("/api/tasks/{tid}/segments/reassign")
async def reassign_segments(tid: str, payload: dict):
    seg_ids = payload.get("segment_ids") or []
    to_speaker = payload.get("to_speaker")
    if not seg_ids or to_speaker is None:
        raise HTTPException(400, "参数非法")
    n = store.reassign_speakers(tid, seg_ids, to_speaker)
    return {"moved_segments": n}


# ---------------------------------------------------------------- 音频回放(含 Range, 流式)

def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].split(",")[0].strip()
    if "-" not in spec:
        return None
    a, b = spec.split("-", 1)
    try:
        if a == "":
            if not b:
                return None
            n = int(b)
            if n <= 0:
                raise ValueError
            start, end = max(0, size - n), size - 1
        else:
            start = int(a)
            end = int(b) if b else size - 1
    except ValueError:
        raise HTTPException(416, "Range 非法")
    if start < 0 or start >= size or end < start:
        raise HTTPException(416, "Range 越界")
    return start, min(end, size - 1)


def _file_stream(path: Path, start: int, end: int, chunk: int = 256 * 1024):
    def gen():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                b = f.read(min(chunk, remaining))
                if not b:
                    break
                remaining -= len(b)
                yield b
    return gen()


@app.get("/api/tasks/{tid}/audio")
async def task_audio(tid: str, request: Request):
    t = store.get_task(tid)
    if not t or not t["audio_path"]:
        raise HTTPException(404, "音频不存在")
    path = Path(t["audio_path"])
    if not path.exists():
        raise HTTPException(404, "音频文件已清理")
    size = path.stat().st_size
    rng = _parse_range(request.headers.get("range") or "", size)
    if rng:
        start, end = rng
        return StreamingResponse(
            _file_stream(path, start, end), status_code=206, media_type="audio/wav",
            headers={"Content-Range": f"bytes {start}-{end}/{size}",
                     "Content-Length": str(end - start + 1),
                     "Accept-Ranges": "bytes"})
    return StreamingResponse(
        _file_stream(path, 0, size - 1), media_type="audio/wav",
        headers={"Content-Length": str(size), "Accept-Ranges": "bytes"})


# ---------------------------------------------------------------- 术语表 / 全局替换

@app.get("/api/terms")
async def list_terms():
    return store.list_terms()


@app.post("/api/terms")
async def add_term(payload: dict):
    wrong = str(payload.get("wrong") or "").strip()
    right = str(payload.get("right") or "").strip()
    if not wrong or not right:
        raise HTTPException(400, "wrong/right 不能为空")
    if wrong == right:
        raise HTTPException(400, "新旧写法相同")
    if len(wrong) > 50 or len(right) > 50:
        raise HTTPException(400, "术语最长 50 字")
    store.add_term(wrong, right)
    return {"wrong": wrong, "right": right}


@app.post("/api/terms/remove")
async def remove_term(payload: dict):
    n = store.delete_term(str(payload.get("wrong") or ""))
    return {"removed": n}


@app.post("/api/tasks/{tid}/replace")
async def replace_text(tid: str, payload: dict):
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    old = str(payload.get("from") or "")
    new = str(payload.get("to") or "")
    if not old:
        raise HTTPException(400, "from 不能为空")
    if old == new:
        raise HTTPException(400, "from 与 to 相同")
    if len(old) > 100 or len(new) > 100:
        raise HTTPException(400, "替换内容最长 100 字")
    if payload.get("dry_run"):
        return store.preview_replace(tid, old, new)
    res = store.apply_replace(tid, old, new)
    if payload.get("save_term") and res["count"]:
        try:
            store.add_term(old, new)
        except ValueError:
            pass
    return res


@app.post("/api/tasks/{tid}/replace/{op_id}/undo")
async def undo_replace(tid: str, op_id: str):
    if not store.get_task(tid):
        raise HTTPException(404, "任务不存在")
    try:
        return store.undo_replace(tid, op_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------- AI 后台任务

def _job_view(job: dict) -> dict:
    out = dict(job)
    for k in ("params", "result"):
        if out.get(k):
            try:
                out[k] = json.loads(out[k])
            except (TypeError, ValueError):
                pass
    return out


def _active_job_or_409(tid: str) -> dict:
    t = store.get_task(tid)
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    return t


def _ensure_no_pipeline(tid: str) -> None:
    if any(j["kind"] == "pipeline"
           for j in store.list_jobs(tid, active_only=True, limit=10)):
        raise HTTPException(409, "一键流水线进行中, 请等待完成或取消")


@app.post("/api/tasks/{tid}/proofread", status_code=202)
async def start_proofread(tid: str, payload: dict | None = None):
    t = _active_job_or_409(tid)
    if t["status"] != "done":
        raise HTTPException(409, "转写未完成")
    _ensure_no_pipeline(tid)
    if not store.get_segments(tid):
        raise HTTPException(409, "无转写片段")
    body = payload or {}
    params = {"only_missing": bool(body.get("only_missing")),
              "force_human": bool(body.get("force_human"))}
    existing = store.find_active_job(tid, "proofread")
    if existing:
        return {"job_id": existing, "deduplicated": True}
    jid = store.create_job(tid, "proofread", params)
    await ai_worker.enqueue(jid)
    return {"job_id": jid, "status": "queued"}


@app.post("/api/tasks/{tid}/understand", status_code=202)
async def start_understand(tid: str):
    _active_job_or_409(tid)
    _ensure_no_pipeline(tid)
    if not store.get_segments(tid):
        raise HTTPException(409, "无转写片段")
    existing = store.find_active_job(tid, "understand")
    if existing:
        return {"job_id": existing, "deduplicated": True}
    jid = store.create_job(tid, "understand", {})
    await ai_worker.enqueue(jid)
    return {"job_id": jid, "status": "queued"}


@app.post("/api/tasks/{tid}/documents", status_code=202)
async def start_document(tid: str, payload: dict):
    _active_job_or_409(tid)
    _ensure_no_pipeline(tid)
    kind = payload.get("kind") or ""
    if kind not in pipeline._DOC_KINDS:
        raise HTTPException(400, f"kind 必须是 {list(pipeline._DOC_KINDS)}")
    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise HTTPException(400, "options 必须是对象")
    if not store.get_documents(tid, "understanding"):
        raise HTTPException(409, "请先生成会议理解(understand)")
    params = {"kind": kind, "options": options}
    existing = store.find_active_job(tid, "document", params)
    if existing:
        return {"job_id": existing, "deduplicated": True}
    jid = store.create_job(tid, "document", params)
    await ai_worker.enqueue(jid)
    return {"job_id": jid, "status": "queued"}


@app.post("/api/tasks/{tid}/pipeline", status_code=202)
async def start_pipeline(tid: str, payload: dict | None = None):
    """一键流水线: 校对(默认只补未校段) → 会议理解 → 指定材料。"""
    t = _active_job_or_409(tid)
    if t["status"] != "done":
        raise HTTPException(409, "转写未完成")
    if not store.get_segments(tid):
        raise HTTPException(409, "无转写片段")
    body = payload or {}
    kind = body.get("kind") or "minutes"
    if kind not in pipeline._DOC_KINDS:
        raise HTTPException(400, f"kind 必须是 {list(pipeline._DOC_KINDS)}")
    options = body.get("options") or {}
    if not isinstance(options, dict):
        raise HTTPException(400, "options 必须是对象")
    existing = store.find_active_job(tid, "pipeline")
    if existing:
        return {"job_id": existing, "deduplicated": True}
    if store.has_active_job(tid):
        raise HTTPException(409, "已有 AI 任务进行中, 请等待完成或取消")
    params = {"kind": kind, "options": options,
              "only_missing": bool(body.get("only_missing", True))}
    jid = store.create_job(tid, "pipeline", params)
    await ai_worker.enqueue(jid)
    return {"job_id": jid, "status": "queued"}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    job = store.get_job(jid)
    if not job:
        raise HTTPException(404, "任务不存在")
    return _job_view(job)


@app.get("/api/tasks/{tid}/jobs")
async def list_jobs(tid: str, kind: str | None = None, active: bool = False, limit: int = 20):
    _active_job_or_409(tid)
    return [_job_view(j) for j in store.list_jobs(tid, kind=kind,
                                                  active_only=active, limit=limit)]


@app.post("/api/jobs/{jid}/cancel")
async def cancel_job(jid: str):
    job = store.get_job(jid)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] == "queued":
        store.update_job(jid, status="cancelled", detail="已取消")
        return {"status": "cancelled"}
    if job["status"] == "running":
        store.update_job(jid, cancel_requested=1)
        return {"status": "cancelling", "note": "当前批次结束后生效"}
    raise HTTPException(409, f"状态 {job['status']} 不可取消")


@app.get("/api/tasks/{tid}/documents")
async def list_documents(tid: str, kind: str | None = None):
    return store.get_documents(tid, kind)


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int):
    d = store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "材料不存在")
    t = store.get_task(d["task_id"])
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    return d


@app.post("/api/documents/{doc_id}/edit")
async def edit_document(doc_id: int, payload: dict):
    """人工编辑材料 → 生成新版本(原版本保留, 历史可追溯)。"""
    d = store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "材料不存在")
    t = store.get_task(d["task_id"])
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    content = str(payload.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "内容不能为空")
    try:
        meta = json.loads(d["meta"] or "{}")
    except (TypeError, ValueError):
        meta = {}
    meta.update({"_edited": True, "_from": doc_id})
    new_id = store.put_document(d["task_id"], d["kind"], content, meta)
    return {"document_id": new_id, "kind": d["kind"], "content": content}


@app.get("/api/documents/{doc_id}/docx")
async def download_docx(doc_id: int):
    d = store.get_document(doc_id)
    if not d:
        raise HTTPException(404, "材料不存在")
    t = store.get_task(d["task_id"])
    if not t or t["deleted"]:
        raise HTTPException(404, "任务不存在")
    kind_name = pipeline._DOC_KINDS.get(d["kind"], (d["kind"], ""))[0]
    title = (t.get("title") or t["filename"] or "会议材料").strip()
    fname = f"{title}-{kind_name}".replace("/", "_").replace("\\", "_") + ".docx"
    data = await asyncio.to_thread(markdown_to_docx, d["content"], fname[:-5])
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"})


# ---------------------------------------------------------------- legacy 兼容

@app.post("/api/transcribe")
async def legacy_transcribe(file: UploadFile = File(...), hotword: str = Form("")):
    """兼容旧接口: 同步转写, 不建任务不留档。新前端请用 /api/tasks。"""
    if not pipeline.asr_ready():
        raise HTTPException(503, pipeline.asr_error() or "模型加载中")
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    token = secrets.token_hex(6)
    tmp_in = Path(config.TMP_DIR) / f"legacy-{token}.bin"
    wav = Path(config.TMP_DIR) / f"legacy-{token}.wav"
    size = 0
    try:
        with open(tmp_in, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"文件超过 {config.MAX_UPLOAD_MB}MB 上限")
                f.write(chunk)
        t0 = time.time()
        async with worker.decode_sem:
            duration = await asyncio.to_thread(pipeline.decode_audio, str(tmp_in), str(wav))
        async with worker.asr_sem:
            segs = await asyncio.to_thread(pipeline.transcribe_wav, str(wav), (hotword or "").strip())
        elapsed = round(time.time() - t0, 1)
        return {"text": "".join(s["raw_text"] for s in segs),
                "segments": [{"start": s["start_ms"], "end": s["end_ms"],
                              "spk": s["speaker_id"], "text": s["raw_text"]} for s in segs],
                "srt": _build_srt(segs), "duration_sec": duration, "elapsed_sec": elapsed}
    finally:
        tmp_in.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


def _fmt_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _build_srt(segments: list[dict]) -> str:
    cues = []
    for s in segments:
        if s["start_ms"] is None:
            continue
        if cues and cues[-1][2] == s["speaker_id"] and (s["end_ms"] - cues[-1][0]) <= 8000 \
                and len(cues[-1][3]) + len(s["raw_text"]) <= 60:
            cues[-1][1] = s["end_ms"]
            cues[-1][3] += s["raw_text"]
        else:
            cues.append([s["start_ms"], s["end_ms"], s["speaker_id"], s["raw_text"]])
    return "\n".join(f"{i}\n{_fmt_ts(a)} --> {_fmt_ts(b)}\n{txt}\n"
                     for i, (a, b, _spk, txt) in enumerate(cues, 1))


@app.post("/api/verify")
async def legacy_verify(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text 为空")
    async with worker.llm_sem:
        try:
            return await asyncio.to_thread(pipeline.verify_text, text,
                                           payload.get("mode", "faithful"),
                                           bool(payload.get("cleanup_filler")))
        except pipeline.LLMError as e:
            raise HTTPException(502, str(e))


# ---------------------------------------------------------------- 静态页

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True),
          name="static")
