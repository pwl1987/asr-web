"""oMLX ASR 后端: HTTP 契约与映射(桩, 不出网)。"""
import pytest

import pipeline
from pipeline import _group_words, transcribe_wav_omlx


class FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._p = payload

    def json(self):
        return self._p


def mkwav(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF")
    return str(p)


def make_post(payload, capture):
    def post(url, data, files, timeout):
        capture["url"] = url
        capture["data"] = dict(data)
        capture["files"] = list(files.keys())
        capture["timeout"] = timeout
        return FakeResp(200, payload)
    return post


def test_omlx_maps_segments(monkeypatch, tmp_path):
    cap: dict = {}
    payload = {"text": "全文", "language": "zh", "duration": 3.5,
               "segments": [{"start": 0.0, "end": 1.2, "text": "大家好。"},
                            {"start": 1.5, "end": 2.4, "text": "今天讲三件事。"}]}
    monkeypatch.setattr(pipeline.config, "ASR_PROVIDER", "omlx")
    segs = transcribe_wav_omlx(mkwav(tmp_path), "灰度 VAD", _post=make_post(payload, cap))
    assert cap["url"].endswith("/v1/audio/transcriptions")
    assert cap["data"]["model"] == pipeline.config.OMLX_ASR_MODEL
    assert cap["data"]["prompt"] == "灰度 VAD"
    assert cap["data"]["response_format"] == "json"
    assert cap["data"]["word_timestamps"] == "true"
    assert "file" in cap["files"]
    assert [(s["start_ms"], s["end_ms"]) for s in segs] == [(0, 1200), (1500, 2400)]
    assert all(s["speaker_id"] is None for s in segs)
    assert [s["raw_text"] for s in segs] == ["大家好。", "今天讲三件事。"]
    assert [s["idx"] for s in segs] == [0, 1]
    # 调度: ASR_PROVIDER=omlx 时 transcribe_wav 走 oMLX 后端
    monkeypatch.setattr(pipeline, "_omlx_post", make_post(payload, cap))
    assert pipeline._real_transcribe_wav(mkwav(tmp_path), "灰度") == segs


def test_omlx_empty_segments_falls_back_to_text(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "ASR_PROVIDER", "omlx")
    segs = transcribe_wav_omlx(mkwav(tmp_path), "",
                               _post=make_post({"text": "只有整段", "duration": 2.0,
                                                "segments": []}, {}))
    assert len(segs) == 1
    assert segs[0]["raw_text"] == "只有整段"
    assert segs[0]["start_ms"] is None


def test_omlx_words_grouping():
    words = [{"text": "大家好", "start": 0.0, "end": 0.6},
             {"text": "今天", "start": 2.0, "end": 2.3},
             {"text": "讲三件事", "start": 2.3, "end": 3.0}]
    g = _group_words(words)
    assert len(g) == 2
    assert g[0]["text"] == "大家好"
    assert g[1]["start"] == 2.0 and g[1]["end"] == 3.0


def test_omlx_http_error(monkeypatch, tmp_path):
    class E:
        status_code = 401

        def json(self):
            return {"detail": "API key required"}

    def post(*a, **k):
        return E()

    monkeypatch.setattr(pipeline.config, "ASR_PROVIDER", "omlx")
    with pytest.raises(RuntimeError, match="401"):
        transcribe_wav_omlx(mkwav(tmp_path), "", _post=post)
