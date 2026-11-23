"""mm-gw-02 ASR Web — FunASR Paraformer 转写 + LLM 校验 (:8466, GPU2)

端点:
  GET  /api/health      服务/模型/ffmpeg 状态
  POST /api/transcribe  multipart: file, hotword? -> {text, segments, srt, duration_sec, elapsed_sec}
  POST /api/verify      json: {text, mode, cleanup_filler} -> {corrected, changes, elapsed_sec}
其余路径静态托管 static/（index.html）。
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ASR_MODEL_DIR", "/data/models/funasr"))
LLM_BASE = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8350/v1")
LLM_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

# ---------------------------------------------------------------- FunASR

_state: dict = {"model": None, "error": None, "load_sec": None, "ffmpeg": None}


def _load_model() -> None:
    try:
        from funasr import AutoModel

        t0 = time.time()
        _state["model"] = AutoModel(
            model=str(MODEL_DIR / "paraformer-zh"),
            vad_model=str(MODEL_DIR / "fsmn-vad"),
            punc_model=str(MODEL_DIR / "ct-punc-zh"),
            spk_model=str(MODEL_DIR / "cam++"),
            disable_update=True,
            disable_pbar=True,
        )
        _state["load_sec"] = round(time.time() - t0, 1)
        _state["error"] = None
    except Exception as e:  # noqa: BLE001 启动期失败要能被 health 看到
        _state["model"] = None
        _state["error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import imageio_ffmpeg

    _state["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_exe()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_model)  # 后台加载, health 可查进度
    yield


app = FastAPI(title="mm-gw-02 asr-web", lifespan=_lifespan)

# ---------------------------------------------------------------- 音频解码

_SUFFIX_BY_TYPE = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/flac": ".flac", "video/webm": ".webm",
}


def _decode_to_wav(data: bytes, content_type: str) -> tuple[str, float, Path]:
    """任意输入 -> 16k mono wav; 返回 (路径, 时长秒)。调用方负责清理目录。"""
    suffix = _SUFFIX_BY_TYPE.get((content_type or "").split(";")[0].strip(), ".bin")
    workdir = Path(tempfile.mkdtemp(prefix="asrweb-"))
    raw, out = workdir / f"in{suffix}", workdir / "out.wav"
    raw.write_bytes(data)
    p = subprocess.run(
        [_state["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(raw), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True, timeout=900,
    )
    if p.returncode != 0 or not out.exists():
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, "音频解码失败: " + p.stderr.decode(errors="replace")[:300])
    duration = max(0.0, (out.stat().st_size - 44) / 2 / 16000)
    return str(out), duration, workdir


# ---------------------------------------------------------------- 转写

def _run_asr(wav_path: str, hotword: str) -> dict:
    res = _state["model"].generate(
        input=wav_path, batch_size_s=300, hotword=hotword or "",
    )
    r = res[0]
    segments = [
        {"start": s["start"], "end": s["end"], "spk": s.get("spk"), "text": s["text"]}
        for s in (r.get("sentence_info") or [])
    ]
    return {"text": (r.get("text") or "").strip(), "segments": segments}


def _fmt_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def build_srt(segments: list[dict]) -> str:
    """同说话人连续句合并为字幕条, 超 8s 或 60 字再切。"""
    cues: list[list] = []  # [start, end, spk, text]
    for s in segments:
        if cues and cues[-1][2] == s["spk"] and (s["end"] - cues[-1][0]) <= 8000 \
                and len(cues[-1][3]) + len(s["text"]) <= 60:
            cues[-1][1] = s["end"]
            cues[-1][3] += s["text"]
        else:
            cues.append([s["start"], s["end"], s["spk"], s["text"]])
    out = []
    for i, (a, b, _spk, txt) in enumerate(cues, 1):
        out.append(f"{i}\n{_fmt_ts(a)} --> {_fmt_ts(b)}\n{txt}\n")
    return "\n".join(out)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    hotword: str = Form(""),
):
    if _state["model"] is None:
        raise HTTPException(503, _state["error"] or "模型加载中，请稍候（约 1 分钟）")
    data = await file.read()
    if len(data) > 512 * 1024 * 1024:
        raise HTTPException(413, "文件超过 512MB 上限")
    wav, duration, workdir = _decode_to_wav(data, file.content_type or "")
    try:
        t0 = time.time()
        result = await asyncio.to_thread(_run_asr, wav, hotword)
        elapsed = round(time.time() - t0, 1)
        result.update(duration_sec=round(duration, 1), elapsed_sec=elapsed,
                      srt=build_srt(result["segments"]))
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- LLM 校验

SYS_FAITHFUL = (
    "你是严谨的中文语音转写校对员。只修正语音识别(ASR)造成的错误，绝不改写原意：\n"
    "1. 同音字/错别字、专名写法；2. 明显的 ASR 幻觉（无中生有的套话如\"谢谢观看\"、机械重复段）；\n"
    "3. 标点与断句。不得增删内容，保留口语用词、方言和原语气。\n"
    "输出严格 JSON：{\"corrected\": \"全文修正稿\", \"changes\": [{\"from\": \"原片段\", \"to\": \"改后\", \"reason\": \"简短原因\"}]}。"
    "changes 只列真实修改处，最多 50 条。"
)
SYS_CLEAN = (
    "你是中文语音转写整理员。在修正 ASR 错误（同音字/错别字/幻觉/标点）的基础上，"
    "去除语气词（嗯、啊、呃、然后、那个等无实义词），清理口语重复与车轱辘话，整理为通顺、可读的书面语，"
    "但不得改变任何事实内容与说话人立场。\n"
    "输出严格 JSON：{\"corrected\": \"整理后全文\", \"changes\": [{\"from\": \"原片段\", \"to\": \"改后\", \"reason\": \"简短原因\"}]}。"
    "changes 只列代表性修改，最多 50 条。"
)
FILLER_NOTE = "额外要求：更彻底地清除语气词与填充词。"


def _strip_fences(s: str) -> str:
    return re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.I)


def _extract_json(s: str) -> dict:
    s = _strip_fences(s)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no json object")
    return json.loads(s[a:b + 1])


@app.post("/api/verify")
async def verify(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text 为空")
    mode = payload.get("mode", "faithful")
    sys_prompt = SYS_CLEAN if mode == "clean" else SYS_FAITHFUL
    if payload.get("cleanup_filler") and mode == "faithful":
        sys_prompt += "\n" + FILLER_NOTE
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "以下是语音转写原文，请按要求处理：\n\n" + text},
        ],
        "temperature": 0.1,
        "max_tokens": max(4096, min(24000, len(text) * 2)),
        "stream": False,
        # huihui 是 thinking 模型, 不关会烧光 max_tokens 且 content 为空
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{LLM_BASE}/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {LLM_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            data = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM 调用失败: {type(e).__name__}: {e}")
    content = data["choices"][0]["message"]["content"]
    if not (content or "").strip():
        raise HTTPException(502, "LLM 返回空 content（thinking 未关闭或输出被截断），请检查 8350 模型")
    try:
        out = _extract_json(content)
        corrected, changes = str(out.get("corrected", "")).strip(), out.get("changes", [])[:80]
    except Exception:  # noqa: BLE001 模型没吐 JSON 就退回原文透传
        corrected, changes = _strip_fences(content), []
    return {"corrected": corrected, "changes": changes,
            "elapsed_sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- 静态页

@app.get("/api/health")
async def health():
    return {
        "model_loaded": _state["model"] is not None,
        "model_error": _state["error"],
        "load_sec": _state["load_sec"],
        "ffmpeg_ok": bool(_state["ffmpeg"]),
        "llm_model": LLM_MODEL,
    }


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
