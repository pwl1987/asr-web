import json, sys, urllib.request
token = open("/data/mm-gw-02/asr-web/data/.auth-token").read().strip()
req = urllib.request.Request(sys.argv[1], headers={"Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req, timeout=30) as r:
    d = json.loads(r.read())
for s in d["segments"][:9]:
    print("spk", s["speaker_id"], "|", s["raw_text"][:42])
