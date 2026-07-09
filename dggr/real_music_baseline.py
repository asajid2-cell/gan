from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf

from gui import backend


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def map_discovered_style_to_codec_genre(style_name: str) -> str:
    text = str(style_name).lower()
    if "noise" in text or "airplane" in text or "white" in text:
        return "cc0_other"
    if "piano" in text or "recitation" in text or "classical" in text:
        return "baroque_classical"
    if "fast" in text or "remix" in text or "percussive" in text:
        return "hiphop_xtc"
    if "jazz" in text or "warm" in text or "tonal" in text:
        return "lofi_hh_lfbb"
    return "cc0_other"


def _split_fixed(audio: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    out: List[np.ndarray] = []
    for start in range(0, len(y), int(chunk_samples)):
        chunk = y[start : start + int(chunk_samples)]
        if len(chunk) < int(chunk_samples):
            chunk = np.pad(chunk, (0, int(chunk_samples) - len(chunk)))
        out.append(chunk.astype(np.float32))
    return out or [np.zeros((int(chunk_samples),), dtype=np.float32)]


def _crossfade_concat(chunks: List[np.ndarray], overlap_samples: int) -> np.ndarray:
    if not chunks:
        return np.zeros((1,), dtype=np.float32)
    out = np.asarray(chunks[0], dtype=np.float32).copy()
    overlap = max(0, int(overlap_samples))
    for raw in chunks[1:]:
        y = np.asarray(raw, dtype=np.float32).reshape(-1)
        n = min(overlap, len(out), len(y))
        if n > 0:
            fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
            fade_in = 1.0 - fade_out
            out = np.concatenate([out[:-n], out[-n:] * fade_out + y[:n] * fade_in, y[n:]])
        else:
            out = np.concatenate([out, y])
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def render_codec_baseline_pack(
    *,
    plan_path: Path,
    out_dir: Path,
    codec_run: str = "run1055",
    codec_checkpoint: Optional[Path] = None,
    max_cases: int = 0,
    seconds: float = 24.0,
    overlap_seconds: float = 0.05,
    device: str = "auto",
    seed: int = 328,
) -> Dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    rows = list(plan.get("rows", []))
    if int(max_cases) > 0:
        rows = rows[: int(max_cases)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = backend.get_codec_session(
        str(codec_run),
        checkpoint_path=str(codec_checkpoint) if codec_checkpoint else None,
        device=str(device),
    )
    sr = int(session.codec.cfg.sample_rate)
    chunk_samples = int(session.codec.target_num_samples())
    want_samples = int(round(float(seconds) * float(sr)))
    overlap_samples = int(round(float(overlap_seconds) * float(sr)))

    out_rows: List[Dict[str, Any]] = []
    for i, case in enumerate(rows):
        case_id = str(case["case_id"])
        source_audio = Path(str(case["source_audio"]))
        real_target = str(case["target_genre"])
        codec_target = map_discovered_style_to_codec_genre(real_target)
        case_dir = out_dir / case_id
        wav_path = case_dir / f"{case_id}__codec_baseline__{codec_target}.wav"
        if not wav_path.exists():
            y, _sr = backend._load_audio_file(source_audio, target_sr=sr)
            clip = y[:want_samples]
            if len(clip) < want_samples:
                clip = np.pad(clip, (0, want_samples - len(clip)))
            generated_chunks: List[np.ndarray] = []
            for chunk_idx, chunk in enumerate(_split_fixed(clip, chunk_samples)):
                result = session.infer_clip(
                    chunk,
                    target_genre=codec_target,
                    style_mode="mix",
                    mix_alpha=0.35,
                    seed=int(seed) + int(i) * 1000 + int(chunk_idx),
                )
                generated_chunks.append(np.asarray(result["generated_audio"], dtype=np.float32))
            wav = _crossfade_concat(generated_chunks, overlap_samples=overlap_samples)
            wav = wav[:want_samples]
            case_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), wav, sr)
        out_rows.append(
            {
                "case_id": case_id,
                "source_audio": str(source_audio),
                "source_genre": str(case.get("source_genre", "")),
                "target_genre": real_target,
                "codec_target_genre": codec_target,
                "seconds": float(seconds),
                "track_id": str(case.get("track_id", "")),
                "generated_wav": str(wav_path),
                "generation_meta": {
                    "baseline": "codec_run1055",
                    "codec_run": str(codec_run),
                    "codec_checkpoint": str(codec_checkpoint) if codec_checkpoint else str(session.record.checkpoint),
                    "codec_target_genre": codec_target,
                    "real_music_target_genre": real_target,
                    "source_audio": str(source_audio),
                    "seconds": float(seconds),
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "codec_baseline_render_done",
                    "case_id": case_id,
                    "done": int(i + 1),
                    "total": int(len(rows)),
                    "codec_target_genre": codec_target,
                    "out_wav": str(wav_path),
                }
            ),
            flush=True,
        )

    manifest = {
        "baseline": "codec_run1055",
        "codec_run": str(codec_run),
        "codec_checkpoint": str(codec_checkpoint) if codec_checkpoint else str(session.record.checkpoint),
        "plan_path": str(plan_path),
        "rows": out_rows,
    }
    _write_json(out_dir / "manifest.json", manifest)
    pd.DataFrame(out_rows).to_csv(out_dir / "manifest.csv", index=False)
    return manifest
