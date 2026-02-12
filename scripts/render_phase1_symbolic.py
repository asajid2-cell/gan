#!/usr/bin/env python
"""
Render Phase 1 symbolic corpora into WAV and emit a manifest for Lab 1.

Inputs:
- Z:/DataSets/_lab1_manifests/pdmx_no_license_conflict_manifest.csv
- Z:/DataSets/_lab1_manifests/the_session_paths.json

Output:
- Z:/DataSets/rendered/phase1_symbolic_audio/**/*.wav
- Z:/DataSets/_lab1_manifests/phase1_symbolic_audio_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pretty_midi
import soundfile as sf

try:
    import music21
except ImportError:  # pragma: no cover
    music21 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render symbolic datasets for Lab 1 Phase 1.")
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=Path(r"Z:/DataSets/_lab1_manifests"),
        help="Directory holding manifest CSV/JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"Z:/DataSets/rendered/phase1_symbolic_audio"),
        help="Directory where rendered WAV files are written.",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=Path(os.environ.get("LAB1_SOUNDFONT", r"Z:/DataSets/soundfonts/MuseScore_General.sf3")),
        help="SoundFont path used by FluidSynth.",
    )
    parser.add_argument("--rate", type=int, default=48000, help="Target sample rate.")
    parser.add_argument("--max-pdmx", type=int, default=1000, help="Max PDMX items to render.")
    parser.add_argument("--max-session", type=int, default=1000, help="Max TheSession tunes to render.")
    parser.add_argument("--seed", type=int, default=328)
    parser.add_argument("--gain", type=float, default=0.7, help="FluidSynth gain.")
    parser.add_argument(
        "--engine",
        choices=["auto", "fluidsynth", "pretty_midi"],
        default="auto",
        help="Render backend. auto prefers FluidSynth when available.",
    )
    parser.add_argument("--force", action="store_true", help="Re-render files that already exist.")
    return parser.parse_args()


def has_fluidsynth() -> bool:
    return shutil.which("fluidsynth") is not None


def safe_name(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not text:
        text = "item"
    return text[:max_len]


def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    rate: int,
    engine: str,
    soundfont: Optional[Path],
    gain: float,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    use_fluidsynth = (
        engine == "fluidsynth"
        or (engine == "auto" and soundfont is not None and soundfont.exists() and has_fluidsynth())
    )

    if use_fluidsynth:
        cmd = [
            "fluidsynth",
            "-ni",
            "-F",
            str(wav_path),
            "-T",
            "wav",
            "-r",
            str(rate),
            "-g",
            str(gain),
            str(soundfont),
            str(midi_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # Fallback path (no external synthesizer required).
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=rate)
    sf.write(str(wav_path), audio, rate, subtype="PCM_16")


def render_pdmx(args: argparse.Namespace, records: List[Dict[str, object]]) -> int:
    pdmx_manifest = args.manifests_root / "pdmx_no_license_conflict_manifest.csv"
    if not pdmx_manifest.exists():
        print(f"[WARN] Missing {pdmx_manifest}; skipping PDMX rendering.")
        return 0

    df = pd.read_csv(pdmx_manifest)
    if "mid_path" not in df.columns:
        print("[WARN] PDMX manifest missing 'mid_path'; skipping PDMX rendering.")
        return 0

    # Keep only rows with existing MIDI files.
    exists_col = "exists_mid_path"
    if exists_col in df.columns:
        df = df[df[exists_col] == True]  # noqa: E712
    else:
        df = df[df["mid_path"].map(lambda p: Path(str(p)).exists())]

    if len(df) == 0:
        print("[WARN] No valid PDMX MIDI rows found after filtering.")
        return 0

    df = df.sample(min(args.max_pdmx, len(df)), random_state=args.seed).reset_index(drop=True)

    rendered = 0
    for idx, row in df.iterrows():
        midi_path = Path(str(row["mid_path"]))
        base = safe_name(midi_path.stem)
        wav_path = args.output_root / "pdmx" / f"{base}.wav"

        if wav_path.exists() and not args.force:
            size = wav_path.stat().st_size
            records.append(
                {"source": "phase1_pdmx", "path": str(wav_path), "ext": ".wav", "size_bytes": int(size), "is_music": 1}
            )
            rendered += 1
            continue

        try:
            render_midi_to_wav(
                midi_path=midi_path,
                wav_path=wav_path,
                rate=args.rate,
                engine=args.engine,
                soundfont=args.soundfont,
                gain=args.gain,
            )
            size = wav_path.stat().st_size
            records.append(
                {"source": "phase1_pdmx", "path": str(wav_path), "ext": ".wav", "size_bytes": int(size), "is_music": 1}
            )
            rendered += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] PDMX render failed ({idx}): {midi_path} :: {exc}")

    return rendered


def make_abc_document(item: Dict[str, object]) -> str:
    name = str(item.get("name", "untitled"))
    meter = str(item.get("meter", "4/4"))
    mode = str(item.get("mode", "Cmajor"))
    abc_body = str(item.get("abc", "")).strip()

    if mode.lower().endswith("major"):
        key = mode[:-5].strip() or "C"
    elif mode.lower().endswith("minor"):
        key = (mode[:-5].strip() or "A") + "m"
    else:
        key = "C"

    return f"X:1\nT:{name}\nM:{meter}\nL:1/8\nK:{key}\n{abc_body}\n"


def render_the_session(args: argparse.Namespace, records: List[Dict[str, object]]) -> int:
    if music21 is None:
        print("[WARN] music21 not installed; skipping TheSession rendering.")
        return 0

    session_paths_file = args.manifests_root / "the_session_paths.json"
    if not session_paths_file.exists():
        print(f"[WARN] Missing {session_paths_file}; skipping TheSession rendering.")
        return 0

    session_paths = json.loads(session_paths_file.read_text(encoding="utf-8"))
    tunes_path = Path(session_paths.get("tunes_json", ""))
    if not tunes_path.exists():
        print(f"[WARN] Missing TheSession tunes json: {tunes_path}")
        return 0

    data = json.loads(tunes_path.read_text(encoding="utf-8"))
    tunes = [x for x in data if str(x.get("abc", "")).strip()]
    if len(tunes) == 0:
        print("[WARN] No TheSession tunes with ABC content found.")
        return 0

    rng = pd.Series(range(len(tunes))).sample(min(args.max_session, len(tunes)), random_state=args.seed).tolist()

    rendered = 0
    for i, tune_idx in enumerate(rng):
        t = tunes[tune_idx]
        tune_id = t.get("tune_id", "na")
        setting_id = t.get("setting_id", "na")
        name = safe_name(str(t.get("name", f"tune_{tune_id}_{setting_id}")))
        base = safe_name(f"{tune_id}_{setting_id}_{name}")
        wav_path = args.output_root / "the_session" / f"{base}.wav"

        if wav_path.exists() and not args.force:
            size = wav_path.stat().st_size
            records.append(
                {
                    "source": "phase1_the_session",
                    "path": str(wav_path),
                    "ext": ".wav",
                    "size_bytes": int(size),
                    "is_music": 1,
                }
            )
            rendered += 1
            continue

        try:
            abc_doc = make_abc_document(t)
            score = music21.converter.parse(abc_doc, format="abc")
            with tempfile.TemporaryDirectory() as td:
                midi_path = Path(td) / "temp.mid"
                score.write("midi", fp=str(midi_path))
                render_midi_to_wav(
                    midi_path=midi_path,
                    wav_path=wav_path,
                    rate=args.rate,
                    engine=args.engine,
                    soundfont=args.soundfont,
                    gain=args.gain,
                )

            size = wav_path.stat().st_size
            records.append(
                {
                    "source": "phase1_the_session",
                    "path": str(wav_path),
                    "ext": ".wav",
                    "size_bytes": int(size),
                    "is_music": 1,
                }
            )
            rendered += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] TheSession render failed ({i}): {base} :: {exc}")

    return rendered


def main() -> None:
    args = parse_args()
    args.manifests_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.engine in ("auto", "fluidsynth"):
        if not has_fluidsynth():
            print("[INFO] fluidsynth not found in PATH; using pretty_midi fallback.")
        elif not args.soundfont.exists():
            print(f"[INFO] SoundFont not found at {args.soundfont}; using pretty_midi fallback.")

    records: List[Dict[str, object]] = []

    n_pdmx = render_pdmx(args, records)
    n_sess = render_the_session(args, records)

    out_csv = args.manifests_root / "phase1_symbolic_audio_manifest.csv"
    out_df = pd.DataFrame(records).drop_duplicates(subset=["path"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    print("\n[DONE] Phase 1 rendering summary")
    print(f"PDMX rows written:        {n_pdmx}")
    print(f"TheSession rows written:  {n_sess}")
    print(f"Manifest rows total:      {len(out_df)}")
    print(f"Manifest path:            {out_csv}")
    if len(out_df):
        print(out_df.groupby("source", as_index=False).agg(files=("path", "count")))


if __name__ == "__main__":
    main()

