"""slycue tests — synthetic tracks with known structure, so cue placement is checkable without real music."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.id3 import ID3, GEOB, TKEY

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slycue  # noqa: E402

SR = 44100
BPM = 128.0
BEAT = 60 / BPM
FIRST = 0.25  # first downbeat, seconds


def make_track(path: Path, sections=((32, 0.08), (64, 0.7), (32, 0.12), (64, 0.75)), root_hz=220.0):
    """Kick on every beat + an A-minor pad; each section = (beats, loudness). Encoded with ffmpeg like a real mp3."""
    beats = sum(b for b, _ in sections)
    n = int((FIRST + beats * BEAT + 2) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    pad = sum(np.sin(2 * np.pi * f * t) for f in (root_hz, root_hz * 1.2, root_hz * 1.5, root_hz * 2))  # A C E A
    b0 = 0
    for count, amp in sections:
        s, e = int((FIRST + b0 * BEAT) * SR), int((FIRST + (b0 + count) * BEAT) * SR)
        y[s:e] += 0.25 * amp * pad[s:e] / 4
        for k in range(count):
            i = int((FIRST + (b0 + k) * BEAT) * SR)
            kt = np.arange(int(0.12 * SR)) / SR
            y[i:i + len(kt)] += amp * np.sin(2 * np.pi * 55 * kt) * np.exp(-kt * 30)
        b0 += count
    wav = path.with_suffix(".wav")
    sf.write(wav, np.clip(y, -1, 1), SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k", str(path)], check=True)
    wav.unlink()
    return path


def serato_grid(path: Path, bpm=BPM, first=FIRST):
    import struct
    try:
        tags = ID3(path)
    except Exception:
        tags = ID3()
    tags.add(GEOB(encoding=0, mime="application/octet-stream", filename="", desc="Serato BeatGrid",
                  data=b"\x01\x00" + struct.pack(">I", 1) + struct.pack(">ff", first, bpm) + b"\x00"))
    tags.save(path)


@pytest.fixture
def track(tmp_path):
    p = make_track(tmp_path / "Artist - Song.mp3")
    serato_grid(p)
    return p


def test_markers2_roundtrip():
    entries = [("COLOR", b"\x00\xff\xff\xff"), ("CUE", slycue.cue_entry(0, 1234, 0xCC0000, "Energy 5"))]
    blob = slycue.dump_markers2(entries)
    assert slycue.parse_markers2(blob) == entries


def test_analyze_puts_cues_on_section_changes(track):
    a = slycue.analyze(track)
    assert abs(a.bpm - BPM) < 0.05 and abs(a.first_beat - FIRST) < 0.01  # Serato grid is trusted
    times = [c.time for c in a.cues]
    assert times[0] == pytest.approx(FIRST, abs=0.01)                      # cue 1 = first downbeat, like MIK
    assert 2 <= len(times) <= 8 and times == sorted(times)
    for boundary_beats in (32, 96, 128):                                    # every real section change gets a cue
        want = FIRST + boundary_beats * BEAT
        assert min(abs(t - want) for t in times) < BEAT / 2, f"no cue near beat {boundary_beats}"
    for t in times:                                                         # all on the 16-beat phrase grid
        assert abs(((t - FIRST) / BEAT) % 16) < 0.01 or abs(((t - FIRST) / BEAT) % 16 - 16) < 0.01
    loud = [c.energy for c in a.cues if abs(c.time - (FIRST + 32 * BEAT)) < 0.1][0]
    quiet = [c.energy for c in a.cues if abs(c.time - (FIRST + 96 * BEAT)) < 0.1][0]
    assert loud > quiet and 1 <= quiet <= 10 and 1 <= loud <= 10


def test_key_detected_when_missing(track):
    assert slycue.analyze(track).camelot == "8A"                            # A minor pad


def test_write_then_skip_then_undo(track, tmp_path):
    tags = ID3(track)
    tags.add(GEOB(encoding=0, mime="application/octet-stream", filename="", desc="Serato Markers2",
                  data=slycue.dump_markers2([("COLOR", b"\x00\xff\xff\xff"), ("BPMLOCK", b"\x00")])))
    tags.save(track)
    log = tmp_path / "undo.jsonl"
    assert slycue.process(track, undo_log=log) == "cued"
    t = ID3(track)
    entries = slycue.parse_markers2([f for f in t.getall("GEOB") if f.desc == "Serato Markers2"][0].data)
    names = [n for n, _ in entries]
    assert names.count("CUE") >= 2 and "COLOR" in names and "BPMLOCK" in names   # existing entries kept
    cue = [d for n, d in entries if n == "CUE"][0]
    assert cue[1] == 0 and cue[12:].split(b"\0")[0].startswith(b"Energy ")   # 00 idx ms*4 00 rgb*3 00 00 name
    assert str(t.getall("TKEY")[0]) == "8A"
    assert slycue.process(track, undo_log=log) == "skip: has cues"              # idempotent
    assert slycue.undo(log) == 1
    t = ID3(track)
    entries = slycue.parse_markers2([f for f in t.getall("GEOB") if f.desc == "Serato Markers2"][0].data)
    assert [n for n, _ in entries] == ["COLOR", "BPMLOCK"] and not t.getall("TKEY")


def test_existing_key_is_never_overwritten(track, tmp_path):
    t = ID3(track); t.add(TKEY(encoding=3, text="11B")); t.save(track)
    slycue.process(track, undo_log=tmp_path / "u.jsonl")
    assert str(ID3(track).getall("TKEY")[0]) == "11B"


def test_refuses_while_serato_runs(track, tmp_path, monkeypatch):
    monkeypatch.setattr(slycue, "serato_running", lambda: True)
    with pytest.raises(SystemExit):
        slycue.main(["run", str(track), "--undo-log", str(tmp_path / "u.jsonl")])
