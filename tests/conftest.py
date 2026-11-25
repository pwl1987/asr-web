"""测试基建: 环境隔离 + ASR/LLM 桩 + 已授权测试客户端。

环境变量在 conftest 顶层设置(先于 app 导入, config 在导入时读 env)。
桩原则: 不依赖真实 GPU / 真实 LLM; decode/transcribe/llm_chat 全部可注入。
"""
import os
import sys
import time
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- 必须在任何 app/config 导入之前 ----
_TMP_DATA = ROOT / ".test-data"
os.environ["ASR_DATA_DIR"] = str(_TMP_DATA)
os.environ["ASR_WEB_TOKEN"] = "test-token-123"
os.environ["ASR_MAX_UPLOAD_MB"] = "2"
os.environ["ASR_MAX_DURATION_MIN"] = "10"
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"  # 桩化后不会真发请求
os.environ["LLM_API_KEY"] = "stub"
os.environ["LLM_MODEL"] = "stub-model"

TOKEN = os.environ["ASR_WEB_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}

from fastapi.testclient import TestClient  # noqa: E402

import pipeline  # noqa: E402


def make_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.fixture(scope="session")
def client():
    # ASR 桩: 模型"已就绪", transcribe 返回 3 段 2 说话人
    pytest.segments_stub = [
        {"idx": 0, "start_ms": 0, "end_ms": 1000, "speaker_id": 0, "raw_text": "今天讨论项目进度"},
        {"idx": 1, "start_ms": 1100, "end_ms": 2000, "speaker_id": 1, "raw_text": "我同意这个方案"},
        {"idx": 2, "start_ms": 2100, "end_ms": 3000, "speaker_id": 0, "raw_text": "那我们定在周五交付"},
    ]
    pipeline._real_decode_audio = pipeline.decode_audio  # 保留真实实现供单元测试
    pipeline.decode_audio = lambda inp, out: (make_wav(Path(out), 1.0) or 1.0)
    pipeline._real_transcribe_wav = pipeline.transcribe_wav  # 保留真实实现供单元测试
    pipeline.transcribe_wav = lambda wav, hotword="": list(pytest.segments_stub)
    pipeline.asr_ready = lambda: True
    pipeline.asr_error = lambda: None
    pipeline.load_asr_model = lambda: None

    from app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def llm_stub(monkeypatch):
    """可编程 LLM 桩: llm_stub.script = [(content, finish_reason), ...] 依次出队;
    未设置 script 时按 prompt 关键词返回各阶段合法输出。"""
    from types import SimpleNamespace
    state = SimpleNamespace(script=[], calls=[])

    def _default(messages):
        sys_c = messages[0]["content"] if messages else ""
        if "校对员" in sys_c:
            return '{"items": []}'
        if "会议分析员" in sys_c:
            return ('{"topic":"测试会议","discussion":[],"viewpoints":[],"disagreements":[]'
                    ',"decisions":[],"suggestions":[],"tasks":[],"unconfirmed":[]}')
        return "# 文档\n内容"

    def fake_llm_chat(messages, max_tokens):
        state.calls.append(messages)
        if state.script:
            return state.script.pop(0)
        return _default(messages), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", fake_llm_chat)
    return state


def wait_task(client, tid, statuses, timeout=10):
    """轮询直到任务进入目标状态集。"""
    if isinstance(statuses, str):
        statuses = {statuses}
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{tid}", headers=AUTH)
        last = r.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout}s 内进入 {statuses}, 最后状态: {last}")


def wait_job(client, jid, statuses=("done",), timeout=10):
    """轮询 AI 后台任务直到目标状态。"""
    if isinstance(statuses, str):
        statuses = {statuses}
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{jid}", headers=AUTH).json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"AI 任务未在 {timeout}s 内进入 {statuses}, 最后: {last}")


def upload(client, name="m.wav", content=b"x" * 1000, hotword=""):
    return client.post("/api/tasks", headers=AUTH,
                       files={"file": (name, content, "audio/wav")},
                       data={"hotword": hotword})


@pytest.fixture(autouse=True)
def _clean_data():
    """每个用例前清空表 + 数据文件(不删 DB 文件, session 级连接仍持有它)。"""
    import shutil

    import app as app_mod
    import config
    for d in (config.UPLOAD_DIR, config.AUDIO_DIR, config.TMP_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    with app_mod.store._lock:
        app_mod.store._conn.executescript(
            "DELETE FROM tasks; DELETE FROM segments; DELETE FROM speakers;"
            " DELETE FROM edits; DELETE FROM documents; DELETE FROM ai_jobs;"
            " DELETE FROM terms; DELETE FROM replace_ops;")
        app_mod.store._conn.commit()
    yield
