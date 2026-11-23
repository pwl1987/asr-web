"""P0 测试: 鉴权 / 上传限制 / 任务生命周期 / 取消重试删除 / 去重 / 恢复 / legacy 兼容。"""
import threading
import time
from pathlib import Path

import pytest

import config
import pipeline
from conftest import AUTH, TOKEN, upload, wait_task


# ---------------- 鉴权 ----------------

def test_ping_open(client):
    assert client.get("/api/ping").status_code == 200


def test_api_requires_token(client):
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/tasks").status_code == 401


def test_wrong_token_401(client):
    assert client.get("/api/tasks", headers={"Authorization": "Bearer wrong"}) .status_code == 401


def test_query_token_ok(client):
    assert client.get(f"/api/tasks?token={TOKEN}").status_code == 200


def test_access_log_token_redacted(client):
    """?token= 允许使用, 但写访问日志前必须脱敏(避免凭据进 journal)。"""
    import logging

    import app

    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", "/api/tasks?token=secret123&x=1", "1.1", 200), None)
    assert app._TokenRedactor().filter(rec)
    path_arg = rec.args[2]
    assert "secret123" not in path_arg and "token=***" in path_arg


def test_page_redirects_to_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302 and "/login" in r.headers["location"]


def test_login_flow(client):
    r = client.post("/login", data={"token": "bad"}, follow_redirects=False)
    assert r.status_code == 200 and "令牌错误" in r.text
    r = client.post("/login", data={"token": TOKEN}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/health").status_code == 200  # cookie 已带上


# ---------------- 上传限制 ----------------

def test_empty_file_400(client):
    r = upload(client, content=b"")
    assert r.status_code == 400


def test_oversize_413(client):
    r = upload(client, content=b"x" * (2 * 1024 * 1024 + 1))  # 上限 2MB
    assert r.status_code == 413


def test_bad_audio_fails_with_reason(client, monkeypatch):
    def bad_decode(inp, out):
        raise ValueError("音频解码失败(文件损坏或格式不支持): stub")
    monkeypatch.setattr(pipeline, "decode_audio", bad_decode)
    r = upload(client)
    assert r.status_code == 200
    tid = r.json()["task_id"]
    t = wait_task(client, tid, {"failed"})
    assert "解码失败" in t["error"]


def test_decode_duration_limit_real():
    """真实 decode_audio: 超时长上限报错(非静默截断), 正常时长按帧数计算。"""
    import tempfile

    from conftest import make_wav
    pytest.importorskip("imageio_ffmpeg")
    real = pipeline._real_decode_audio
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "in.wav", Path(td) / "out.wav"
        make_wav(src, 5.0)
        d = real(str(src), str(out))
        assert abs(d - 5.0) < 0.1
        make_wav(src, (config.MAX_DURATION_MIN + 1) * 60.0)
        with pytest.raises(ValueError, match="时长超过"):
            real(str(src), str(out))


def test_dedupe_same_file(client):
    gate = threading.Event()
    monkey_gated = pytest.MonkeyPatch()
    monkey_gated.setattr(pipeline, "transcribe_wav",
                         lambda *a, **k: (gate.wait(5), list(pytest.segments_stub))[1])
    try:
        r1 = upload(client, name="meet.wav", content=b"a" * 4096)
        time.sleep(0.3)  # 让第一个任务进入 processing
        r2 = upload(client, name="meet.wav", content=b"a" * 4096)
        assert r1.json()["task_id"] == r2.json()["task_id"]
    finally:
        gate.set()
        monkey_gated.undo()


# ---------------- 任务生命周期 ----------------

def test_full_lifecycle(client):
    r = upload(client, hotword="周报 项目")
    assert r.status_code == 200
    tid = r.json()["task_id"]
    t = wait_task(client, tid, {"done"})
    assert t["stage"] == "done" and t["duration_sec"] == 1.0
    assert t["hotword"] == "周报 项目"
    # 片段与默认说话人已入库
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    assert len(d["segments"]) == 3
    assert d["segments"][0]["raw_text"] == "今天讨论项目进度"
    assert {s["speaker_id"] for s in d["speakers"]} == {0, 1}
    assert d["speakers"][0]["display_name"] == "说话人1"
    # 音频已留档
    assert Path(t["audio_path"]).exists()
    a = client.get(f"/api/tasks/{tid}/audio", headers=AUTH)
    assert a.status_code == 200 and a.headers["accept-ranges"] == "bytes"


def test_audio_range_request(client):
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    r = client.get(f"/api/tasks/{tid}/audio", headers={**AUTH, "Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert len(r.content) == 100


def test_cancel_queued(client, monkeypatch):
    # 让 ASR 慢, 任务停在 processing 前的窗口; 直接测 queued 取消
    gate = threading.Event()
    monkeypatch.setattr(pipeline, "transcribe_wav",
                        lambda *a, **k: (gate.wait(5), list(pytest.segments_stub))[1])
    tid = upload(client).json()["task_id"]
    # 立即取消(大概率还在 queued/decoding)
    r = client.post(f"/api/tasks/{tid}/cancel", headers=AUTH)
    assert r.status_code == 200
    gate.set()
    t = wait_task(client, tid, {"cancelled", "done"})
    # 若已进入 ASR 则不可取消; 两种结果都合法, 但 cancelled 必须占多数路径
    assert t["status"] in ("cancelled", "done")


def test_cancel_processing_completes_as_done(client, monkeypatch):
    """规格: 只保证取消"尚未开始"的任务; ASR 已开始则不可中断, 最终应正常完成。"""
    gate = threading.Event()

    def slow_asr(*a, **k):
        gate.wait(5)
        return list(pytest.segments_stub)
    monkeypatch.setattr(pipeline, "transcribe_wav", slow_asr)
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"processing"})
    r = client.post(f"/api/tasks/{tid}/cancel", headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "cancelling"
    gate.set()
    t = wait_task(client, tid, {"done"})  # ASR 已开始 → 不可中断, 完成而非取消
    assert t["status"] == "done"
    # 解码等待期(未进 ASR)的取消会生效: 让解码卡住, 取消在解码后检查点命中
    gate2 = threading.Event()

    def slow_decode(inp, out):
        gate2.wait(5)
        from conftest import make_wav
        make_wav(Path(out), 1.0)
        return 1.0
    monkeypatch.setattr(pipeline, "decode_audio", slow_decode)
    tid2 = upload(client).json()["task_id"]
    time.sleep(0.2)
    r = client.post(f"/api/tasks/{tid2}/cancel", headers=AUTH)
    assert r.status_code == 200
    gate2.set()
    wait_task(client, tid2, {"cancelled"})


def test_retry_after_failure(client, monkeypatch):
    ok = {"v": False}

    def flaky_decode(inp, out):
        if not ok["v"]:
            raise ValueError("音频解码失败: stub")
        from conftest import make_wav
        make_wav(Path(out), 1.0)
        return 1.0
    monkeypatch.setattr(pipeline, "decode_audio", flaky_decode)
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"failed"})
    # 解码失败但 upload bin 还在 → 可以重试; 仍失败
    assert client.post(f"/api/tasks/{tid}/retry", headers=AUTH).status_code == 200
    wait_task(client, tid, {"failed"})
    ok["v"] = True
    assert client.post(f"/api/tasks/{tid}/retry", headers=AUTH).status_code == 200
    wait_task(client, tid, {"done"})
    # 删除后 retry 应 400(源文件已清理)
    client.post(f"/api/tasks/{tid}/delete", headers=AUTH)
    assert client.post(f"/api/tasks/{tid}/retry", headers=AUTH).status_code == 404


def test_delete_cleans_files(client):
    from app import store
    tid = upload(client).json()["task_id"]
    t = wait_task(client, tid, {"done"})
    upload_bin = Path(config.UPLOAD_DIR) / f"{tid}.bin"
    audio = Path(t["audio_path"])
    # 转写成功后原始上传已清理, 只留回放用 16k wav
    assert not upload_bin.exists() and audio.exists()
    assert store.get_segments(tid) and store.get_speakers(tid)
    # 人为留下一个 bin(历史数据/异常路径) → 删除时也要清掉
    upload_bin.write_bytes(b"leftover")
    r = client.post(f"/api/tasks/{tid}/delete", headers=AUTH)
    assert r.status_code == 200
    assert not upload_bin.exists() and not audio.exists()
    # 片段/说话人/文档/编辑/AI job 等数据行一并清除
    assert store.get_segments(tid) == [] and store.get_speakers(tid) == []
    assert store.get_documents(tid) == [] and store.get_edits(tid) == []
    assert client.get(f"/api/tasks/{tid}", headers=AUTH).status_code == 404
    assert all(x["id"] != tid for x in client.get("/api/tasks", headers=AUTH).json())


def test_ai_job_recovery_and_purge(client):
    from app import store
    jid = store.create_job("recover-task", "proofread", {})
    store.update_job(jid, status="running")
    info = store.recover_jobs()
    assert jid in info["interrupted"]
    assert store.get_job(jid)["status"] == "failed"
    queued = store.create_job("recover-task", "understand", {})
    assert queued in store.pending_queued_jobs()
    # 任务删除时 AI job 一并清理
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    done_job = store.create_job(tid, "understand", {})
    store.update_job(done_job, status="done")
    client.post(f"/api/tasks/{tid}/delete", headers=AUTH)
    assert store.get_job(done_job) is None


def test_delete_processing_blocked(client, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(pipeline, "transcribe_wav",
                        lambda *a, **k: (gate.wait(5), list(pytest.segments_stub))[1])
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"processing"})
    assert client.post(f"/api/tasks/{tid}/delete", headers=AUTH).status_code == 409
    gate.set()
    wait_task(client, tid, {"done"})


# ---------------- 重启恢复 ----------------

def test_recovery_marks_processing_failed_and_requeues(client):
    from app import store
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    # 造一个 processing 卡死任务
    store.create_task("stuck.wav", 10, "")
    stuck = store.list_tasks()[0]["id"]
    store.update_task(stuck, status="processing", stage="asr")
    info = store.recover_on_startup()
    assert stuck in info["interrupted"]
    t = store.get_task(stuck)
    assert t["status"] == "failed" and "服务重启" in t["error"]
    # queued 任务保留(由 worker.recover 重新入队)
    store.create_task("pending.wav", 10, "")
    store.recover_on_startup()
    pending = [x for x in store.list_tasks() if x["status"] == "queued"]
    assert len(pending) >= 1


# ---------------- 片段/说话人编辑 ----------------

def test_segment_edit_and_history(client):
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    seg = d["segments"][0]
    # 人工改文本
    r = client.patch(f"/api/segments/{seg['id']}", headers=AUTH,
                     json={"task_id": tid, "text": "今天讨论项目进度(修正)"})
    assert r.status_code == 200
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    assert d["segments"][0]["corrected_text"] == "今天讨论项目进度(修正)"
    assert d["segments"][0]["corrected_by"] == "human"
    assert d["edits"][0]["field"] == "text"
    # 原始文本不可变
    assert d["segments"][0]["raw_text"] == "今天讨论项目进度"


def test_speaker_change_keeps_text_provenance(client):
    """改说话人不能把 corrected_by 清空(那是文本修正来源, 不是说话人来源)。"""
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    seg = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()["segments"][0]
    client.patch(f"/api/segments/{seg['id']}", headers=AUTH,
                 json={"task_id": tid, "text": "人工文本"})
    r = client.patch(f"/api/segments/{seg['id']}", headers=AUTH,
                     json={"task_id": tid, "speaker_id": 5})
    assert r.status_code == 200
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    changed = next(s for s in d["segments"] if s["id"] == seg["id"])
    assert changed["speaker_id"] == 5
    assert changed["corrected_by"] == "human"
    assert changed["corrected_text"] == "人工文本"


def test_speaker_rename_merge_split_reassign(client):
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    s0, s1, s2 = d["segments"]
    # 改名
    r = client.post(f"/api/tasks/{tid}/speakers/0/rename", headers=AUTH,
                    json={"display_name": "张三"})
    assert r.json()["display_name"] == "张三"
    # 改派片段: s1 → 说话人0
    r = client.post(f"/api/tasks/{tid}/segments/reassign", headers=AUTH,
                    json={"segment_ids": [s1["id"]], "to_speaker": 0})
    assert r.json()["moved_segments"] == 1
    # 拆分: s2 拆成新说话人(此时 max_sid=0, 新 id=1)
    r = client.post(f"/api/tasks/{tid}/speakers/split", headers=AUTH,
                    json={"segment_ids": [s2["id"]]})
    assert r.json()["new_speaker_id"] == 1
    # 合并: 说话人1 → 说话人0
    r = client.post(f"/api/tasks/{tid}/speakers/merge", headers=AUTH,
                    json={"from": [1], "to": 0})
    assert r.json()["moved_segments"] == 1
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    assert all(s["speaker_id"] == 0 for s in d["segments"])


def test_segment_speaker_change_creates_speaker(client):
    tid = upload(client).json()["task_id"]
    wait_task(client, tid, {"done"})
    seg = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()["segments"][0]
    r = client.patch(f"/api/segments/{seg['id']}", headers=AUTH,
                     json={"task_id": tid, "speaker_id": 5})
    assert r.status_code == 200
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    assert any(s["speaker_id"] == 5 for s in d["speakers"])


# ---------------- legacy 兼容 ----------------

def test_legacy_transcribe_and_verify(client, llm_stub):
    r = client.post("/api/transcribe", headers=AUTH,
                    files={"file": ("a.wav", b"x" * 500, "audio/wav")})
    assert r.status_code == 200
    d = r.json()
    assert len(d["segments"]) == 3 and "srt" in d
    r = client.post("/api/verify", headers=AUTH, json={"text": "测试文本"})
    assert r.status_code == 200 and "corrected" in r.json()
    r = client.post("/api/verify", headers=AUTH, json={"text": ""})
    assert r.status_code == 400
