"""集中配置: 环境变量 + 资源限制。全部可被环境覆盖, 便于测试注入。"""
import os
from pathlib import Path

def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ASR_DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "asrweb.db"
UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"
TMP_DIR = DATA_DIR / "tmp"
TOKEN_FILE = DATA_DIR / ".auth-token"

MODEL_DIR = Path(os.environ.get("ASR_MODEL_DIR", "/data/models/funasr"))

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8350/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
# thinking 模型(如 huihui/Qwen3)必须关闭, 否则 token 烧进 reasoning_content
LLM_DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "1") == "1"

# 鉴权: 未设置时启动自动生成并持久化到 TOKEN_FILE(0600)
ASR_WEB_TOKEN = os.environ.get("ASR_WEB_TOKEN", "")

# 限制
MAX_UPLOAD_MB = _int("ASR_MAX_UPLOAD_MB", 512)
MAX_DURATION_MIN = _int("ASR_MAX_DURATION_MIN", 120)
FFMPEG_TIMEOUT = _int("ASR_FFMPEG_TIMEOUT", 900)          # 秒
LLM_TIMEOUT = _int("ASR_LLM_TIMEOUT", 1200)               # 秒
DECODE_CONCURRENCY = _int("ASR_DECODE_CONCURRENCY", 2)
ASR_CONCURRENCY = _int("ASR_ASR_CONCURRENCY", 1)          # 同一张 GPU, 串行
LLM_CONCURRENCY = _int("ASR_LLM_CONCURRENCY", 2)          # 全局限流(线程级)
AI_JOB_CONCURRENCY = _int("ASR_AI_JOB_CONCURRENCY", 2)    # 同时运行的 AI 任务数
PROOFREAD_WORKERS = _int("ASR_PROOFREAD_WORKERS", 0) or LLM_CONCURRENCY
STAGE_A_BATCH_CHARS = _int("ASR_STAGE_A_BATCH_CHARS", 2500)
UNDERSTAND_CHUNK_CHARS = _int("ASR_UNDERSTAND_CHUNK_CHARS", 12000)
UNDERSTAND_MAX_TOKENS = _int("ASR_UNDERSTAND_MAX_TOKENS", 12000)
MODEL_WAIT_SEC = _int("ASR_MODEL_WAIT_SEC", 180)          # 启动后任务等待模型就绪上限
MAX_OUTPUT_TOKENS_CAP = _int("ASR_MAX_OUTPUT_TOKENS", 24000)
# 转写成功后是否保留原始上传(默认 0: 只留 16k wav, 避免双份占盘)
KEEP_UPLOAD = os.environ.get("ASR_KEEP_UPLOAD", "0") == "1"

def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, AUDIO_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
