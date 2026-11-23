"""P1 测试: AI 后台任务(校对/理解/材料)、人工修改保护、分批判错、取消与补缺。"""
import json
import threading
import time

from conftest import AUTH, upload, wait_job, wait_task


def _done_task(client):
    tid = upload(client, hotword="周五 项目").json()["task_id"]
    wait_task(client, tid, {"done"})
    return tid


def _segs(client, tid):
    return client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()


def _start(client, tid, kind, body=None):
    r = client.post(f"/api/tasks/{tid}/{kind}", headers=AUTH, json=body or {})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def _understanding(client, tid):
    docs = client.get(f"/api/tasks/{tid}/documents?kind=understanding", headers=AUTH).json()
    assert docs
    return json.loads(docs[0]["content"])


# ---------------- 阶段 A: 忠实校对 ----------------

def test_proofread_job_missing_id_retry(client, monkeypatch):
    import pipeline

    tid = _done_task(client)
    calls = {"n": 0}

    def flaky_llm(messages, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:  # 第一次漏掉 id=1
            return json.dumps({"items": [
                {"id": 0, "text": "今天讨论项目进度(校)"},
                {"id": 2, "text": "那我们定在周五交付(校)"}]}), "stop"
        return json.dumps({"items": [{"id": 1, "text": "我同意这个方案(校)"}]}), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", flaky_llm)
    jid = _start(client, tid, "proofread")
    j = wait_job(client, jid)
    assert j["result"]["corrected_count"] == 3
    d = _segs(client, tid)
    assert d["segments"][1]["corrected_text"] == "我同意这个方案(校)"
    assert d["segments"][1]["corrected_by"] == "llm"
    assert d["segments"][0]["raw_text"] == "今天讨论项目进度"  # 原始结果不可变


def test_proofread_all_batches_fail_marks_job_failed(client, monkeypatch):
    import pipeline

    tid = _done_task(client)
    monkeypatch.setattr(pipeline, "llm_chat",
                        lambda messages, max_tokens: ("这不是JSON", "stop"))
    jid = _start(client, tid, "proofread")
    j = wait_job(client, jid, {"failed"})
    assert "JSON" in j["error"]


def test_proofread_requires_done(client):
    r = client.post("/api/tasks/nonexistent/proofread", headers=AUTH)
    assert r.status_code == 404


def test_proofread_never_overwrites_human_edit(client, monkeypatch):
    import pipeline

    tid = _done_task(client)
    d = _segs(client, tid)
    seg0 = d["segments"][0]
    client.patch(f"/api/segments/{seg0['id']}", headers=AUTH,
                 json={"task_id": tid, "text": "人工定稿文本"})

    def llm(messages, max_tokens):
        return json.dumps({"items": [
            {"id": 0, "text": "LLM想改的文本"},
            {"id": 1, "text": "我同意这个方案(校)"},
            {"id": 2, "text": "那我们定在周五交付(校)"}]}), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", llm)
    jid = _start(client, tid, "proofread")
    j = wait_job(client, jid)
    d = _segs(client, tid)
    assert d["segments"][0]["corrected_text"] == "人工定稿文本"
    assert d["segments"][0]["corrected_by"] == "human"
    assert d["segments"][1]["corrected_by"] == "llm"
    assert j["result"]["stats"]["skipped_human"] == 1
    assert j["result"]["corrected_count"] == 2


def test_proofread_partial_batches_persist(client, monkeypatch):
    import config
    import pipeline

    tid = _done_task(client)
    monkeypatch.setattr(config, "STAGE_A_BATCH_CHARS", 10)
    monkeypatch.setattr(config, "PROOFREAD_WORKERS", 1)
    calls = {"n": 0}

    def flaky(messages, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"items": [{"id": 0, "text": "甲(校)"},
                                         {"id": 1, "text": "乙(校)"}]}), "stop"
        raise pipeline.LLMError("上游超时(stub)")

    monkeypatch.setattr(pipeline, "llm_chat", flaky)
    jid = _start(client, tid, "proofread")
    j = wait_job(client, jid)
    assert j["status"] == "done"  # 部分成功仍算完成, 但明确报告缺段
    assert j["result"]["corrected_count"] == 2
    assert j["result"]["remaining_missing"] == 1
    assert j["result"]["stats"]["failed_batches"] >= 1
    d = _segs(client, tid)
    assert d["segments"][0]["corrected_text"] == "甲(校)"

    # 补缺: 只处理未校对段, 已校对的 LLM 结果不重跑
    calls["n"] = 0
    monkeypatch.setattr(pipeline, "llm_chat",
                        lambda messages, max_tokens: (json.dumps(
                            {"items": [{"id": 2, "text": "丙(校)"}]}), "stop"))
    jid2 = _start(client, tid, "proofread", {"only_missing": True})
    j2 = wait_job(client, jid2)
    assert j2["result"]["corrected_count"] == 1
    assert j2["result"]["remaining_missing"] == 0


def test_proofread_cancel_keeps_finished_batches(client, monkeypatch):
    import config
    import pipeline

    tid = _done_task(client)
    monkeypatch.setattr(config, "STAGE_A_BATCH_CHARS", 10)
    monkeypatch.setattr(config, "PROOFREAD_WORKERS", 1)
    gate = threading.Event()
    calls = {"n": 0}

    def slow(messages, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            gate.wait(5)
            return json.dumps({"items": [{"id": 0, "text": "甲(校)"},
                                         {"id": 1, "text": "乙(校)"}]}), "stop"
        return json.dumps({"items": [{"id": 2, "text": "丙(校)"}]}), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", slow)
    jid = _start(client, tid, "proofread")
    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/api/jobs/{jid}", headers=AUTH).json()["status"] == "running":
            break
        time.sleep(0.05)
    r = client.post(f"/api/jobs/{jid}/cancel", headers=AUTH)
    assert r.status_code == 200
    gate.set()
    j = wait_job(client, jid, {"cancelled"})
    assert j["result"]["corrected_count"] >= 1
    assert j["result"]["stats"]["cancelled"] is True
    d = _segs(client, tid)
    assert d["segments"][0]["corrected_text"] == "甲(校)"


def test_active_job_dedupe_and_task_delete_guard(client, monkeypatch):
    """重复点击复用同一 job; 有 AI 任务时任务不可删除。"""
    import pipeline

    tid = _done_task(client)
    gate = threading.Event()

    def slow(messages, max_tokens):
        gate.wait(5)
        return json.dumps({"items": []}), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", slow)
    jid1 = _start(client, tid, "proofread")
    r2 = client.post(f"/api/tasks/{tid}/proofread", headers=AUTH, json={})
    assert r2.status_code == 202 and r2.json()["deduplicated"] is True
    assert r2.json()["job_id"] == jid1
    jobs = client.get(f"/api/tasks/{tid}/jobs?active=1", headers=AUTH).json()
    assert [j["id"] for j in jobs] == [jid1]
    assert client.post(f"/api/tasks/{tid}/delete", headers=AUTH).status_code == 409
    client.post(f"/api/jobs/{jid1}/cancel", headers=AUTH)
    gate.set()
    wait_job(client, jid1, {"cancelled", "done"})


# ---------------- 阶段 B: 会议理解 ----------------

def test_understand_filters_invalid_segment_refs(client, llm_stub):
    tid = _done_task(client)
    llm_stub.script = [(json.dumps({
        "topic": "项目周会",
        "discussion": [{"title": "进度", "summary": "正常", "segment_ids": [0, 9999]}],
        "viewpoints": [{"speaker": "张三", "point": "同意", "segment_ids": [1]}],
        "disagreements": [], "decisions": [{"decision": "周五交付", "owner": "张三",
                                            "deadline": "周五", "segment_ids": [2]}],
        "suggestions": [], "tasks": [{"content": "提交周报", "owner": "李四",
                                      "deadline": "周五", "segment_ids": [1, "bad"]}],
        "unconfirmed": []}, ensure_ascii=False), "stop")]
    jid = _start(client, tid, "understand")
    j = wait_job(client, jid)
    u = _understanding(client, tid)
    docs = client.get(f"/api/tasks/{tid}/documents", headers=AUTH).json()
    assert j["result"]["document_id"] in [d["id"] for d in docs]
    assert u["discussion"][0]["segment_ids"] == [0]        # 9999 被过滤
    assert u["discussion"][0]["invalid_refs"] == 1
    assert u["tasks"][0]["segment_ids"] == [1]             # 非法类型被过滤
    assert u["topic"] == "项目周会"


def test_understand_truncated_output_splits_and_merges(client, llm_stub):
    tid = _done_task(client)
    left = {"topic": "前半段", "discussion": [],
            "decisions": [{"decision": "决定A", "owner": "张三", "deadline": "",
                           "segment_ids": [0]}],
            "viewpoints": [], "disagreements": [], "suggestions": [], "tasks": [],
            "unconfirmed": []}
    right = {"topic": "", "discussion": [],
             "decisions": [{"decision": "决定B", "owner": "李四", "deadline": "",
                            "segment_ids": [1, 2]}],
             "viewpoints": [], "disagreements": [], "suggestions": [], "tasks": [],
             "unconfirmed": []}
    llm_stub.script = [("{}", "length"),
                       (json.dumps(left, ensure_ascii=False), "stop"),
                       (json.dumps(right, ensure_ascii=False), "stop")]
    jid = _start(client, tid, "understand")
    j = wait_job(client, jid)
    assert j["status"] == "done"
    u = _understanding(client, tid)
    assert u["topic"] == "前半段"
    assert [x["decision"] for x in u["decisions"]] == ["决定A", "决定B"]


def test_understand_stored_as_document(client, llm_stub):
    tid = _done_task(client)
    jid = _start(client, tid, "understand")
    j = wait_job(client, jid)
    docs = client.get(f"/api/tasks/{tid}/documents", headers=AUTH).json()
    assert any(d["kind"] == "understanding" for d in docs)
    assert j["result"]["document_id"] in [d["id"] for d in docs]


# ---------------- 阶段 C: 材料生成 ----------------

def test_document_requires_understanding(client, llm_stub):
    tid = _done_task(client)
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "minutes", "options": {}})
    assert r.status_code == 409  # 未生成理解


def test_document_kinds_and_options(client, llm_stub):
    tid = _done_task(client)
    jid = _start(client, tid, "understand")
    wait_job(client, jid)
    llm_stub.script = [("# 会议纪要\n- 决定: 周五交付", "stop")]
    jid = _start(client, tid, "documents", {
        "kind": "minutes", "options": {
            "title": "项目周会纪要", "audience": "分管领导",
            "word_count": 500, "include_names": True}})
    j = wait_job(client, jid)
    assert "周五交付" in j["result"]["content"]
    sys_msg = llm_stub.calls[-1][0]["content"]
    assert "分管领导" in sys_msg and "500" in sys_msg
    # 非法 kind
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "poem", "options": {}})
    assert r.status_code == 400


def test_document_truncation_continuation(client, llm_stub):
    """finish_reason=length → 自动续写拼接; 仍截断则标注。"""
    tid = _done_task(client)
    jid = _start(client, tid, "understand")
    wait_job(client, jid)
    llm_stub.script = [("第一部分", "length"), ("第二部分", "stop")]
    jid = _start(client, tid, "documents", {"kind": "outline", "options": {}})
    j = wait_job(client, jid)
    assert j["result"]["content"] == "第一部分第二部分"

    # 两次续写后仍截断 → 附截断标注
    llm_stub.script = [("A", "length"), ("B", "length"), ("C", "length")]
    jid = _start(client, tid, "documents", {"kind": "outline", "options": {}})
    j = wait_job(client, jid)
    assert "截断" in j["result"]["content"]


# ---------------- legacy /api/verify(整理模式) ----------------

def test_verify_clean_mode_uses_clean_prompt(client, llm_stub):
    llm_stub.script = [(json.dumps({"corrected": "干净文本", "changes": []},
                                   ensure_ascii=False), "stop")]
    r = client.post("/api/verify", headers=AUTH,
                    json={"text": "嗯那个测试", "mode": "clean", "cleanup_filler": True})
    assert r.status_code == 200
    sys_msg = llm_stub.calls[0][0]["content"]
    assert "整理" in sys_msg  # clean 提示词生效
