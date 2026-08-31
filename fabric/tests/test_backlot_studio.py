"""backlot studio lanes: speak/transcribe/stems/export/mesh/stepseq."""

import base64
import json
from unittest.mock import patch

from fabric import backlot


def _serve(rows, bodies=None):
    seq = list(rows)
    captured = bodies if bodies is not None else []

    def _open(req, timeout=0):
        if req.data:
            captured.append(json.loads(req.data.decode()))
        code, payload = seq.pop(0)

        class R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                if isinstance(code, int) and code != 200:
                    return json.dumps({"detail": "err"}).encode()
                return json.dumps(payload).encode()

        return R()

    return _open, captured


def test_speak_saves_wav(tmp_path):
    wav = b"RIFF" + b"\x00" * 64
    data_url = "data:audio/wav;base64," + base64.b64encode(wav).decode()
    open_, _ = _serve([(200, {"audio": data_url})])
    with patch.object(backlot, "_SPEAK_DIR", str(tmp_path)), \
            patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.speak("hello ship")
    assert out["ok"] is True
    assert out["bytes"] == len(wav)
    with open(out["file"], "rb") as f:
        assert f.read() == wav


def test_transcribe_encodes_data_url(tmp_path):
    src = tmp_path / "clip.wav"
    raw = b"WAVEdata123"
    src.write_bytes(raw)
    open_, bodies = _serve([(200, {"text": "hello", "language": "en",
                                   "segments": [{"t": 0}]})])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.transcribe(str(src))
    assert out["ok"] is True and out["text"] == "hello"
    sent = bodies[0]["audio"]
    assert sent.startswith("data:audio/wav;base64,")
    assert base64.b64decode(sent.split(",", 1)[1]) == raw


def test_stems_submit_returns_job(tmp_path):
    src = tmp_path / "song.mp3"
    src.write_bytes(b"mp3-bytes")
    open_, bodies = _serve([(200, {"job_id": 9})])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.stems_submit(str(src), model="htdemucs_ft")
    assert out["job_id"] == 9 and out["status"] == "pending"
    assert bodies[0]["model"] == "htdemucs_ft"


def test_sample_export_validates_format():
    out = backlot.sample_export("zip", ["a"])
    assert "wavset|volca|ableton" in out["error"]
    out2 = backlot.sample_export("volca", [])
    assert "non-empty" in out2["error"]


def test_stepseq_requires_prompt_and_steps():
    assert "error" in backlot.stepseq({"steps": [1]})
    assert "error" in backlot.stepseq({"prompt": "x"})


def test_mesh_upgrade_int_check():
    assert "error" in backlot.mesh_upgrade("nope")


def test_transcode_quality_validation():
    assert "error" in backlot.transcode_submit("/x.mp4", quality=99)
    assert "error" in backlot.transcode_submit("")


def test_transcode_parses_response():
    open_, bodies = _serve([(200, {"output_path": "/o/x.av1.mp4",
                                   "codec": "av1", "size_bytes": 123,
                                   "fallback": False})])
    with patch.object(backlot.urllib.request, "urlopen", open_):
        out = backlot.transcode_submit("x.mp4", quality=35)
    assert out["codec"] == "av1" and out["fallback"] is False
