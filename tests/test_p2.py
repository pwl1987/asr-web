"""P2 测试: 一键流水线 / 术语与全局替换 / 标题搜索 / 材料编辑与 DOCX / 引用清洗。"""
import io
import json

from conftest import AUTH, upload, wait_job, wait_task


def _done_task(client):
    tid = upload(client, hotword="周五 项目").json()["task_id"]
    wait_task(client, tid, {"done"})
    return tid


def _start(client, tid, path, body=None):
    r = client.post(f"/api/tasks/{tid}/{path}", headers=AUTH, json=body or {})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


# ---------------- 标题 / 搜索 ----------------

def test_task_title_and_search(client):
    tid = _done_task(client)
    r = client.patch(f"/api/tasks/{tid}", headers=AUTH, json={"title": "预算评审会"})
    assert r.status_code == 200 and r.json()["title"] == "预算评审会"
    assert client.patch(f"/api/tasks/{tid}", headers=AUTH,
                        json={"title": " "}).status_code == 400

    hits = client.get("/api/tasks?q=预算", headers=AUTH).json()
    assert [t["id"] for t in hits] == [tid]
    # 搜转写内容 → 带命中片段
    hits = client.get("/api/tasks?q=周五", headers=AUTH).json()
    assert hits and hits[0]["id"] == tid
    assert "周五" in hits[0]["match_snippet"]
    assert client.get("/api/tasks?q=不存在的词", headers=AUTH).json() == []


# ---------------- 术语 / 全局替换 ----------------

def test_global_replace_preview_apply_undo(client):
    tid = _done_task(client)
    pre = client.post(f"/api/tasks/{tid}/replace", headers=AUTH,
                      json={"from": "项目", "to": "工程", "dry_run": True}).json()
    assert pre["count"] == 1 and pre["examples"][0]["idx"] == 0

    res = client.post(f"/api/tasks/{tid}/replace", headers=AUTH,
                      json={"from": "项目", "to": "工程", "save_term": True}).json()
    assert res["count"] == 1 and res["op_id"]
    d = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()
    seg0 = d["segments"][0]
    assert seg0["corrected_text"] == "今天讨论工程进度"
    assert seg0["corrected_by"] == "human"
    assert seg0["raw_text"] == "今天讨论项目进度"  # 原始结果不可变
    assert any(t["wrong"] == "项目" and t["right"] == "工程"
               for t in client.get("/api/terms", headers=AUTH).json())

    undo = client.post(f"/api/tasks/{tid}/replace/{res['op_id']}/undo", headers=AUTH).json()
    assert undo["restored"] == 1
    seg0 = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()["segments"][0]
    assert seg0["corrected_text"] is None and seg0["corrected_by"] is None


def test_glossary_applied_after_proofread(client, monkeypatch):
    import pipeline

    tid = _done_task(client)
    client.post("/api/terms", headers=AUTH, json={"wrong": "张山", "right": "张三"})

    def llm(messages, max_tokens):
        return json.dumps({"items": [
            {"id": 0, "text": "今天讨论项目进度"},
            {"id": 1, "text": "张山负责周五交付"},
            {"id": 2, "text": "那我们定在周五交付"}]}, ensure_ascii=False), "stop"

    monkeypatch.setattr(pipeline, "llm_chat", llm)
    jid = _start(client, tid, "proofread")
    wait_job(client, jid)
    segs = client.get(f"/api/tasks/{tid}/segments", headers=AUTH).json()["segments"]
    assert "张三" in segs[1]["corrected_text"]
    assert "张山" not in segs[1]["corrected_text"]


# ---------------- 一键流水线 ----------------

def test_pipeline_end_to_end_and_citation_cleanup(client, llm_stub):
    tid = _done_task(client)
    items = {"items": [
        {"id": 0, "text": "今天讨论项目进度(校)"},
        {"id": 1, "text": "我同意这个方案(校)"},
        {"id": 2, "text": "那我们定在周五交付(校)"}]}
    understanding = {"topic": "流水线会议", "discussion": [],
                     "viewpoints": [], "disagreements": [],
                     "decisions": [{"decision": "周五交付", "owner": "张三",
                                    "deadline": "周五", "segment_ids": [2]}],
                     "suggestions": [], "tasks": [], "unconfirmed": []}
    doc_md = "# 流水线纪要\n\n- 决定: 周五交付 [#2]\n- 假引用 [#999]\n"
    llm_stub.script = [(json.dumps(items, ensure_ascii=False), "stop"),
                       (json.dumps(understanding, ensure_ascii=False), "stop"),
                       (doc_md, "stop")]
    jid = _start(client, tid, "pipeline",
                 {"kind": "minutes", "options": {"title": "流水线纪要"}})
    j = wait_job(client, jid, timeout=20)
    assert j["status"] == "done"
    assert j["result"]["proofread"]["corrected_count"] == 3
    doc = j["result"]["document"]
    assert "[#2]" in doc["content"] and "[#999]" not in doc["content"]
    assert doc["citations"] == 1
    # 自动用会议主题命名任务
    assert client.get(f"/api/tasks/{tid}", headers=AUTH).json()["title"] == "流水线会议"
    kinds = [d["kind"] for d in client.get(f"/api/tasks/{tid}/documents", headers=AUTH).json()]
    assert "understanding" in kinds and "minutes" in kinds


# ---------------- 材料编辑 / DOCX ----------------

def test_document_edit_creates_version_and_docx(client, llm_stub):
    tid = _done_task(client)
    jid = _start(client, tid, "pipeline", {"kind": "minutes", "options": {}})
    j = wait_job(client, jid, timeout=20)
    old_id = j["result"]["document"]["document_id"]
    old = client.get(f"/api/documents/{old_id}", headers=AUTH).json()

    r = client.post(f"/api/documents/{old_id}/edit", headers=AUTH,
                    json={"content": "# 修改版\n\n- 决定 A"})
    assert r.status_code == 200
    new_id = r.json()["document_id"]
    assert new_id != old_id
    new = client.get(f"/api/documents/{new_id}", headers=AUTH).json()
    assert "修改版" in new["content"]
    assert json.loads(new["meta"])["_edited"] is True
    # 旧版本保留
    assert client.get(f"/api/documents/{old_id}", headers=AUTH).json()["content"] == old["content"]

    x = client.get(f"/api/documents/{new_id}/docx", headers=AUTH)
    assert x.status_code == 200
    assert x.content[:2] == b"PK"
    assert "wordprocessingml" in x.headers["content-type"]
    assert ".docx" in x.headers["content-disposition"]


def test_docx_markdown_basic():
    from docx import Document

    from docx_export import markdown_to_docx

    data = markdown_to_docx("# 标题\n\n- 要点 **重点**\n\n| 列1 | 列2 |\n| --- | --- |\n| a | b |",
                            "标题")
    assert data[:2] == b"PK"
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "标题" in text and "重点" in text
    assert doc.tables and doc.tables[0].cell(1, 1).text == "b"
