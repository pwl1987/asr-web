"""真实端到端评估: 合成 3 人会议 → 任务转写 → CER/说话人 → 校对 → 理解 → 纪要。
用法: python3 e2e_eval.py <wav> <script.json> <token>
在 H3C 服务器上运行(需访问 127.0.0.1:8466 与 8350)。
"""
import json
import sys
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8466"
wav_path, script_path, token = sys.argv[1], sys.argv[2], sys.argv[3]


def api(method: str, path: str, data=None, files=None, timeout=1200, hotword=None):
    url = BASE + path
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if files:
        boundary = uuid.uuid4().hex
        parts = []
        if hotword:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f"name=\"hotword\"\r\n\r\n{hotword}\r\n")
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                     f"filename=\"{files[0]}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
        post = f"\r\n--{boundary}--\r\n"
        body = ("".join(parts).encode() + files[1] + post.encode())
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif data is not None:
        body = json.dumps(data, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_job(path: str, body=None) -> str:
    return api("POST", path, data=body or {})["job_id"]


def wait_job(jid: str) -> dict:
    deadline = time.time() + 1800
    while time.time() < deadline:
        j = api("GET", f"/api/jobs/{jid}")
        if j["status"] in ("done", "failed", "cancelled"):
            assert j["status"] == "done", j
            return j
        time.sleep(2)
    raise TimeoutError(jid)


def cer(ref: str, hyp: str) -> float:
    """字级 CER(去标点空白后)。"""
    import re
    def norm(s):
        return re.sub(r"[\s，。！？、；：\"\"''《》（）,.!?;:()\-—…·]", "", s)
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


t0 = time.time()
script = json.load(open(script_path))
ref_text = "".join(x["text"] for x in script)
ref_speakers = len({x["speaker"] for x in script})

wav_bytes = open(wav_path, "rb").read()
up = api("POST", "/api/tasks", files=("meeting.wav", wav_bytes),
         hotword=(sys.argv[4] if len(sys.argv) > 4 else None))
tid = up["task_id"]
print(f"[upload] task={tid}")

deadline = time.time() + 600
while True:
    t = api("GET", f"/api/tasks/{tid}")
    if t["status"] in ("done", "failed"):
        break
    time.sleep(2)
assert t["status"] == "done", t
asr_elapsed = time.time() - t0
print(f"[asr] done: duration={t['duration_sec']}s wall={asr_elapsed:.1f}s")

d = api("GET", f"/api/tasks/{tid}/segments")
segs = d["segments"]
hyp_text = "".join(s["raw_text"] for s in segs)
raw_cer = cer(ref_text, hyp_text)
n_spk = len({s["speaker_id"] for s in segs if s["speaker_id"] is not None})
print(f"[asr] segments={len(segs)} speakers_detected={n_spk} (真实 {ref_speakers}) RAW_CER={raw_cer:.4f}")

# 校对(后台 job)
p = wait_job(start_job(f"/api/tasks/{tid}/proofread"))["result"]
print(f"[proofread] corrected={p['corrected_count']} missing={p['stats']['missing_ids']} "
      f"failed_batches={p['stats']['failed_batches']}")
d2 = api("GET", f"/api/tasks/{tid}/segments")
corr_text = "".join(s["corrected_text"] or s["raw_text"] for s in d2["segments"])
corr_cer = cer(ref_text, corr_text)
print(f"[proofread] CORR_CER={corr_cer:.4f} (raw {raw_cer:.4f})")

# 理解(后台 job) → 取最新 understanding 文档
j = wait_job(start_job(f"/api/tasks/{tid}/understand"))
docs = api("GET", f"/api/tasks/{tid}/documents?kind=understanding")
u = json.loads(docs[0]["content"])
print(f"[understand] topic={u['topic']}")
print(f"  decisions={len(u['decisions'])} tasks={len(u['tasks'])} "
      f"disagreements={len(u['disagreements'])} unconfirmed={len(u['unconfirmed'])}")
for x in u["decisions"]:
    print(f"    决定: {x['decision']} owner={x['owner']} refs={x['segment_ids']}")
for x in u["tasks"]:
    print(f"    待办: {x['content']} owner={x['owner']} deadline={x['deadline']} refs={x['segment_ids']}")

# 纪要(后台 job)
doc = wait_job(start_job(f"/api/tasks/{tid}/documents",
                         {"kind": "minutes", "options": {"title": "项目例会纪要",
                                                         "include_names": True}}))["result"]
print(f"[minutes] {len(doc['content'])} chars")
total_elapsed = time.time() - t0
print(f"[total] wall={total_elapsed:.1f}s (音频 {t['duration_sec']}s, 实时率 {t['duration_sec']/total_elapsed:.1f}x)")

json.dump({"task_id": tid, "raw_cer": raw_cer, "corr_cer": corr_cer,
           "segments": len(segs), "speakers": n_spk,
           "duration": t["duration_sec"], "wall": total_elapsed},
          open("/tmp/e2e_result.json", "w"), ensure_ascii=False)
print("[saved] /tmp/e2e_result.json")
