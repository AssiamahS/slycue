#!/usr/bin/env python3
"""slycue — Mixed In Key-style cue points for a Serato library, written into the files themselves.

For every track that has no hot cues yet it:
  * takes the beatgrid Serato already wrote (falls back to librosa beat tracking),
  * scores every 16-beat phrase for energy (loudness + onset density) and places up to 8 hot cues where
    the energy changes — cue 1 on the first downbeat, all cues on the phrase grid, named "Energy N" like MIK,
  * detects the key when the file has none (Camelot, MIK style) and adds EnergyLevel + "8A - Energy 6".

Cues go into the "Serato Markers2" GEOB frame, so Serato, DJ5ly and anything else that reads Serato tags
sees them. Existing cues, keys and comments are never overwritten. Every write is logged to an undo log.

    slycue scan ~/Music                 # how many tracks have cues / don't (tags only, fast)
    slycue run ~/Music/yt-dlp           # cue every track that has none (Serato must be closed)
    slycue run FILE --dry-run           # print the cues without writing
    slycue undo                         # put every changed frame back the way it was
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from mutagen.aiff import AIFF
from mutagen.id3 import COMM, GEOB, ID3, TKEY, TXXX, ID3NoHeaderError

EXTS = {".mp3", ".aif", ".aiff"}
PHRASE = 16                      # beats per phrase; MIK puts cues on 16-beat boundaries
MIN_GAP = 16                     # beats between two cues (MIK often cues consecutive phrases)
MAX_CUES = 8
MIK_COLORS = [0xCC0000, 0xCC8800, 0x0000CC, 0xCCCC00, 0x00CC00, 0xCC00CC, 0x00CCCC, 0x8800CC]
DEFAULT_UNDO = Path.home() / ".slycue" / "undo.jsonl"
NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CAMELOT_MAJ = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MIN = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]
KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


@dataclass
class Cue:
    time: float
    energy: int


@dataclass
class Analysis:
    bpm: float
    first_beat: float
    cues: list[Cue] = field(default_factory=list)
    camelot: str | None = None
    energy: int = 0


# ------------------------------------------------------------------ Serato Markers2
def cue_entry(index: int, ms: int, color: int, name: str) -> bytes:
    """00, index, u32 ms, 00, rgb, 00 00, name NUL — the layout Serato and MIK write."""
    return b"\x00" + bytes([index]) + struct.pack(">I", ms) + b"\x00" + color.to_bytes(3, "big") + b"\x00\x00" + name.encode() + b"\x00"


def parse_markers2(blob: bytes) -> list[tuple[str, bytes]]:
    text = blob[2:].split(b"\x00")[0].replace(b"\n", b"")
    raw = base64.b64decode(text + b"=" * (-len(text) % 4))
    out, o = [], 2
    while o < len(raw):
        e = raw.find(b"\x00", o)
        if e <= o or e + 5 > len(raw):
            break
        n = struct.unpack(">I", raw[e + 1:e + 5])[0]
        out.append((raw[o:e].decode("latin1"), raw[e + 5:e + 5 + n]))
        o = e + 5 + n
    return out


def dump_markers2(entries: list[tuple[str, bytes]]) -> bytes:
    data = b"\x01\x01" + b"".join(n.encode() + b"\x00" + struct.pack(">I", len(d)) + d for n, d in entries) + b"\x00"
    return b"\x01\x01" + base64.b64encode(data)        # same shape MIK writes (verified byte-for-byte round trip)


# ------------------------------------------------------------------ tags
def load_tags(path: Path):
    """(container, ID3) — for AIFF the container saves; for mp3 the ID3 object saves itself."""
    if path.suffix.lower() in (".aif", ".aiff"):
        f = AIFF(path)
        if f.tags is None:
            f.add_tags()
        return f, f.tags
    try:
        t = ID3(path)
    except ID3NoHeaderError:
        t = ID3()
    return t, t


def save_tags(path: Path, container, tags):
    v = 3 if tags.version[:2] == (2, 3) else 4
    if container is tags:
        tags.save(path, v2_version=v)
    else:
        container.save(v2_version=v)


def markers2(tags) -> GEOB | None:
    return next((f for f in tags.getall("GEOB") if f.desc == "Serato Markers2"), None)


def has_cues(tags) -> bool:
    m = markers2(tags)
    try:
        return bool(m) and any(n == "CUE" for n, _ in parse_markers2(m.data))
    except Exception:
        return False


def serato_grid(tags) -> tuple[float, float] | None:
    g = next((f for f in tags.getall("GEOB") if f.desc == "Serato BeatGrid"), None)
    if not g or len(g.data) < 14:
        return None
    count = struct.unpack(">I", g.data[2:6])[0]
    if not count or 6 + count * 8 > len(g.data):
        return None
    pos0 = struct.unpack(">f", g.data[6:10])[0]
    if count == 1:
        bpm = struct.unpack(">f", g.data[10:14])[0]
    else:
        beats = struct.unpack(">I", g.data[10:14])[0]
        bpm = beats * 60 / (struct.unpack(">f", g.data[14:18])[0] - pos0)
    return (bpm, max(0.0, pos0)) if 40 < bpm < 250 else None


# ------------------------------------------------------------------ analysis
def detect_key(y, sr) -> str:
    import librosa
    import numpy as np
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096).mean(axis=1)
    best = (-2.0, 0, False)
    for r in range(12):
        rolled = np.roll(chroma, -r)
        for prof, minor in ((KS_MAJ, False), (KS_MIN, True)):
            c = float(np.corrcoef(rolled, prof)[0, 1])
            if c > best[0]:
                best = (c, r, minor)
    return (CAMELOT_MIN if best[2] else CAMELOT_MAJ)[best[1]]


def analyze(path: Path, want_key: bool = True) -> Analysis:
    import librosa
    import numpy as np
    _, tags = load_tags(path)
    y, sr = librosa.load(path, sr=22050, mono=True)
    dur = len(y) / sr
    grid = serato_grid(tags)
    if grid:
        bpm, first = grid
    else:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm, first = float(np.atleast_1d(tempo)[0]), float(beats[0]) if len(beats) else 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    fps = sr / hop
    step = PHRASE * 60 / bpm
    starts = [first + i * step for i in range(int((dur - first) / step))]
    feats = []
    for s in starts:
        a, b = int(s * fps), int((s + step) * fps)
        db = 20 * np.log10(np.mean(rms[a:b]) + 1e-9)
        feats.append((db, float(np.mean(onset[a:b]))))
    if not feats:
        return Analysis(bpm, first, [Cue(first, 1)])
    dbs = np.array([f[0] for f in feats])
    ons = np.array([f[1] for f in feats]); ons = ons / (np.percentile(ons, 95) + 1e-9)
    # energy 1-10: absolute loudness (-36..-6 dBFS RMS) weighted with relative rhythmic density
    levels = np.clip(np.round(1 + 9 * (0.75 * np.clip((dbs + 36) / 30, 0, 1) + 0.25 * np.clip(ons, 0, 1))), 1, 10).astype(int)
    change = np.abs(np.diff(dbs)) + 3 * np.abs(np.diff(ons))            # score for the boundary before phrase i+1
    picked = [0]
    order = sorted(range(1, len(starts)), key=lambda i: -change[i - 1])
    for strict in (True, False):                                       # real changes first, then fill like MIK does
        for i in order:
            if len(picked) >= MAX_CUES:
                break
            if strict and change[i - 1] < 1.5:
                continue
            if i not in picked and all(abs(i - j) * PHRASE >= MIN_GAP for j in picked):
                picked.append(i)
    cues = [Cue(round(starts[i], 4), int(levels[i])) for i in sorted(picked)]
    key = detect_key(y, sr) if want_key else None
    return Analysis(bpm, first, cues, key, int(np.clip(round(np.percentile(levels, 75)), 1, 10)))


# ------------------------------------------------------------------ write / undo
def write(path: Path, a: Analysis, undo_log: Path) -> None:
    container, tags = load_tags(path)
    m = markers2(tags)
    keep = [(n, d) for n, d in parse_markers2(m.data) if n != "CUE"] if m else [("COLOR", b"\x00\xff\xff\xff"), ("BPMLOCK", b"\x00")]
    cues = [("CUE", cue_entry(i, int(round(c.time * 1000)), MIK_COLORS[i], f"Energy {c.energy}")) for i, c in enumerate(a.cues[:MAX_CUES])]
    record = {"path": str(path), "markers2": base64.b64encode(m.data).decode() if m else None, "added": [], "t": time.time()}
    tags.delall("GEOB:Serato Markers2")
    tags.add(GEOB(encoding=0, mime="application/octet-stream", filename="", desc="Serato Markers2", data=dump_markers2(keep + cues)))
    if a.camelot and not tags.getall("TKEY"):
        tags.add(TKEY(encoding=3, text=a.camelot)); record["added"].append("TKEY")
    if a.energy and not tags.getall("TXXX:EnergyLevel"):
        tags.add(TXXX(encoding=3, desc="EnergyLevel", text=str(a.energy))); record["added"].append("TXXX:EnergyLevel")
    key = a.camelot or (str(tags.getall("TKEY")[0]) if tags.getall("TKEY") else "")
    if a.energy and key and not tags.getall("COMM"):
        tags.add(COMM(encoding=3, lang="eng", desc="", text=f"{key} - Energy {a.energy}")); record["added"].append("COMM::eng")
    save_tags(path, container, tags)
    undo_log.parent.mkdir(parents=True, exist_ok=True)
    with undo_log.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def process(path: Path, undo_log: Path = DEFAULT_UNDO, dry_run: bool = False) -> str:
    _, tags = load_tags(path)
    if has_cues(tags):
        return "skip: has cues"
    a = analyze(path, want_key=not tags.getall("TKEY"))
    if dry_run:
        return "dry: " + ", ".join(f"{c.time:.2f}s E{c.energy}" for c in a.cues) + (f" key {a.camelot}" if a.camelot else "")
    write(path, a, undo_log)
    return "cued"


def undo(undo_log: Path = DEFAULT_UNDO) -> int:
    if not undo_log.exists():
        return 0
    records = [json.loads(line) for line in undo_log.read_text().splitlines() if line.strip()]
    done = 0
    for r in reversed(records):
        p = Path(r["path"])
        if not p.exists():
            continue
        container, tags = load_tags(p)
        tags.delall("GEOB:Serato Markers2")
        if r["markers2"]:
            tags.add(GEOB(encoding=0, mime="application/octet-stream", filename="", desc="Serato Markers2", data=base64.b64decode(r["markers2"])))
        for k in r["added"]:
            tags.delall(k)
        save_tags(p, container, tags)
        done += 1
    undo_log.rename(undo_log.with_suffix(f".undone-{int(time.time())}.jsonl"))
    return done


# ------------------------------------------------------------------ CLI
def serato_running() -> bool:
    return subprocess.run(["pgrep", "-f", "/Serato DJ"], capture_output=True).returncode == 0


def audio_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        p = p.expanduser()
        if p.is_file():
            out.append(p)
        else:
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS and not f.name.startswith("._") and "_Serato_" not in f.parts)
    return out


def _work(args):
    path, log, dry = args
    try:
        return path, process(path, log, dry)
    except Exception as e:  # one broken file must not stop a 5,000-track run
        return path, f"error: {type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slycue", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("paths", nargs="+")
    r = sub.add_parser("run"); r.add_argument("paths", nargs="+")
    r.add_argument("--dry-run", action="store_true"); r.add_argument("--limit", type=int)
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    r.add_argument("--undo-log", type=Path, default=DEFAULT_UNDO); r.add_argument("--force", action="store_true", help="write even if Serato is open")
    u = sub.add_parser("undo"); u.add_argument("--undo-log", type=Path, default=DEFAULT_UNDO)
    a = ap.parse_args(argv)

    if a.cmd == "undo":
        print(f"restored {undo(a.undo_log)} files"); return
    files = audio_files(a.paths)
    if a.cmd == "scan":
        missing = [f for f in files if not has_cues(load_tags(f)[1])]
        print(f"{len(files)} tracks · {len(files) - len(missing)} have cues · {len(missing)} without")
        return
    if not a.dry_run and not a.force and serato_running():
        sys.exit("Serato is open — quit Serato first (it rewrites cue tags of loaded tracks), or pass --force.")
    todo = [f for f in files if not has_cues(load_tags(f)[1])][: a.limit]
    print(f"{len(files)} tracks, {len(todo)} without cues → cueing with {a.jobs} workers", flush=True)
    t0, counts = time.time(), {}
    with ProcessPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(_work, (f, a.undo_log, a.dry_run)) for f in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            path, res = fut.result()
            counts[res.split(":")[0]] = counts.get(res.split(":")[0], 0) + 1
            if a.dry_run or res.startswith("error") or n % 25 == 0 or n == len(todo):
                eta = (time.time() - t0) / n * (len(todo) - n)
                print(f"[{n}/{len(todo)} eta {eta / 60:.0f}m] {path.name[:60]} — {res}", flush=True)
    print("done:", counts, f"in {(time.time() - t0) / 60:.1f} min · undo log {a.undo_log}")


if __name__ == "__main__":
    main()
