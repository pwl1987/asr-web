"""P1 阶段测试: LLM 分阶段(校对/理解/材料)的结构校验、漏段重试、截断续写、权限。"""
import json

from conftest import AUTH, upload, wait_task


def _done_task(client):
    tid = upload(client, hotword="周五 项目").json()["task_id"]
    wait_task(client, tid, {"done"})
    return tid


def _segs(client, tid):
    return client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()


# ---------------- 阶段 A: 忠实校对 ----------------

def test_proofread_missing_id_retry(client, llm_stub, monkeypatch):
    import pipeline

    tid = _done_task(client)
    real_llm = pipeline.llm_chat
    calls = {"n": 0}

    def flaky_llm(messages, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:  # 第一次漏掉 id=1
            return json.dumps({"items": [
                {"id": 0, "text": "今天讨论项目进度(校)"}, {"id": 2, "text": "那我们定在周五交付(校)"}]}), "stop"
        return json.dumps({"items": [{"id": 1, "text": "我同意这个方案(校)"}]}), "stop"
    monkeypatch.setattr(pipeline, "llm_chat", flaky_llm)

    r = client.post(f"/api/tasks/{tid}/proofread", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["corrected_count"] == 3
    d = _segs(client, tid)
    assert d["segments"][1]["corrected_text"] == "我同意这个方案(校)"
    assert d["segments"][1]["corrected_by"] == "llm"
    assert d["segments"][0]["raw_text"] == "今天讨论项目进度"  # 原始结果不可变


def test_proofread_malformed_json_502(client, monkeypatch):
    import pipeline

    tid = _done_task(client)
    monkeypatch.setattr(pipeline, "llm_chat",
                        lambda messages, max_tokens: ("这不是JSON", "stop"))
    r = client.post(f"/api/tasks/{tid}/proofread", headers=AUTH)
    assert r.status_code == 502


def test_proofread_requires_done(client):
    # 没有任务 → 404
    r = client.post("/api/tasks/nonexistent/proofread", headers=AUTH)
    assert r.status_code == 404


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
    r = client.post(f"/api/tasks/{tid}/understand", headers=AUTH)
    assert r.status_code == 200
    u = r.json()["understanding"]
    assert u["discussion"][0]["segment_ids"] == [0]        # 9999 被过滤
    assert u["discussion"][0]["invalid_refs"] == 1
    assert u["tasks"][0]["segment_ids"] == [1]             # 非法类型被过滤
    assert u["topic"] == "项目周会"


def test_understand_stored_as_document(client, llm_stub):
    tid = _done_task(client)
    client.post(f"/api/tasks/{tid}/understand", headers=AUTH)
    docs = client.get(f"/api/tasks/{tid}/documents", headers=AUTH).json()
    assert any(d["kind"] == "understanding" for d in docs)


# ---------------- 阶段 C: 材料生成 ----------------

def test_document_requires_understanding(client, llm_stub):
    tid = _done_task(client)
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "minutes", "options": {}})
    assert r.status_code == 409  # 未生成理解


def test_document_kinds_and_options(client, llm_stub):
    tid = _done_task(client)
    client.post(f"/api/tasks/{tid}/understand", headers=AUTH)
    llm_stub.script = [("# 会议纪要\n- 决定: 周五交付", "stop")]
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "minutes", "options": {
                        "title": "项目周会纪要", "audience": "分管领导",
                        "word_count": 500, "include_names": True}})
    assert r.status_code == 200
    assert "周五交付" in r.json()["content"]
    # 调用参数里应带上选项
    sys_msg = llm_stub.calls[-1][0]["content"]
    assert "分管领导" in sys_msg and "500" in sys_msg
    # 非法 kind
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "poem", "options": {}})
    assert r.status_code == 400


def test_document_truncation_continuation(client, llm_stub, monkeypatch):
    """finish_reason=length → 自动续写拼接; 仍截断则标注。"""
    tid = _done_task(client)
    client.post(f"/api/tasks/{tid}/understand", headers=AUTH)
    llm_stub.script = [("第一部分", "length"), ("第二部分", "stop")]
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "outline", "options": {}})
    assert r.status_code == 200
    assert r.json()["content"] == "第一部分第二部分"

    # 两次续写后仍截断 → 附截断标注
    llm_stub.script = [("A", "length"), ("B", "length"), ("C", "length")]
    r = client.post(f"/api/tasks/{tid}/documents", headers=AUTH,
                    json={"kind": "outline", "options": {}})
    assert "截断" in r.json()["content"]


# ---------------- legacy /api/verify(整理模式) ----------------

def test_verify_clean_mode_uses_clean_prompt(client, llm_stub):
    llm_stub.script = [(json.dumps({"corrected": "干净文本", "changes": []},
                                   ensure_ascii=False), "stop")]
    r = client.post("/api/verify", headers=AUTH,
                    json={"text": "嗯那个测试", "mode": "clean", "cleanup_filler": True})
    assert r.status_code == 200
    sys_msg = llm_stub.calls[0][0]["content"]
    assert "整理" in sys_msg  # clean 提示词生效
