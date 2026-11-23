"""生成 3 人合成会议录音(Mac say TTS) + 台词 JSON(供 CER/说话人核对)。
用法: python3 gen_meeting.py /tmp/meeting  ->  meeting.wav + meeting_script.json
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/meeting")
OUT.mkdir(parents=True, exist_ok=True)

LINES = [
    ("张伟", "Eddy (中文（中国大陆）)",
     "呃，各位下午好，现在开始本周的项目例会。首先请李婷汇报一下线上支付故障的排查进展。"),
    ("李婷", "Flo (中文（中国大陆）)",
     "好的。昨天夜里两点的告警是网关超时引起的，我们已经把有问题的节点摘除了，流量切到备用集群，今晚会观察复发情况。另外我想申请把告警阈值从五秒调到三秒。"),
    ("张伟", "Eddy (中文（中国大陆）)",
     "这个可以，运维那边你去对接。接下来请王强讲一下新版本灰度的计划。"),
    ("王强", "Grandpa (中文（中国大陆）)",
     "新版本会在周四开始灰度，先放百分之五的流量，如果投诉不上升，周五再放到百分之二十。风险主要是老接口的兼容性，我们已经准备了回滚预案。"),
    ("李婷", "Flo (中文（中国大陆）)",
     "我有不同意见，我觉得百分之二十太激进了，建议先放到百分之十，观察一天再说。"),
    ("王强", "Grandpa (中文（中国大陆）)",
     "可以接受，那就先百分之十，看一天的数据再决定。"),
    ("张伟", "Eddy (中文（中国大陆）)",
     "好，那就按这个执行。记录一下会议决定：一，告警阈值由李婷本周内调整到三秒；二，新版本周四开始灰度百分之十，由王强负责；三，故障复盘文档周五下班前提交给我。"),
    ("李婷", "Flo (中文（中国大陆）)",
     "没有问题。"),
    ("王强", "Grandpa (中文（中国大陆）)",
     "复盘文档我会前把初稿发到群里，大家有时间先看一下。"),
    ("张伟", "Eddy (中文（中国大陆）)",
     "今天的内容就到这里，散会。"),
]

def tts(text: str, voice: str, dst: Path) -> None:
    aiff = dst.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(dst)], check=True)
    aiff.unlink()

wavs = []
for i, (spk, voice, text) in enumerate(LINES):
    p = OUT / f"line{i:02d}.wav"
    tts(text, voice, p)
    wavs.append(p)

sil = b"\x00\x00" * 8000  # 0.5s 静音(16k mono)
with wave.open(str(OUT / "meeting.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(sil)
    for p in wavs:
        with wave.open(str(p), "rb") as r:
            w.writeframes(r.readframes(r.getnframes()))
        w.writeframes(sil)

(OUT / "meeting_script.json").write_text(json.dumps(
    [{"speaker": s, "text": t} for s, _, t in LINES], ensure_ascii=False, indent=1),
    encoding="utf-8")
import wave as _w
with _w.open(str(OUT / "meeting.wav")) as w:
    print(f"meeting.wav: {w.getnframes()/w.getframerate():.1f}s")
