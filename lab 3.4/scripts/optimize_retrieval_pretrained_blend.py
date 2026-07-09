from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB33_SCRIPTS = REPO_ROOT / "lab 3.3" / "scripts"
if str(LAB33_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB33_SCRIPTS))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dggr.lab3_diffusion_data import DIFFUSION_SR
from run_hybrid_vocal_push_compare import HybridPushConfig, _resolve_stems, picked_songs
from train_scratch_retrieval_fusion import _device_from_arg, _judge_probs_for_audio, _load_or_train_judge, _mix_preserved_vocals, _slug, _write_json
from train_scratch_structure_diffusion import _audio_metrics


def _read_manifest(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["source_song"]), str(row["target"]))
            rows[key] = {str(k): str(v) for k, v in row.items()}
    return rows


def _match_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float32), np.asarray(b[:n], dtype=np.float32)


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y / peak * 0.999
    return y.astype(np.float32)


def _body_inject(base: np.ndarray, donor: np.ndarray, *, alpha: float, low_hz: float = 180.0, high_hz: float = 3200.0) -> np.ndarray:
    base, donor = _match_length(base, donor)
    n_fft = 2048
    hop = 512
    base_stft = librosa.stft(base, n_fft=n_fft, hop_length=hop)
    donor_stft = librosa.stft(donor, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=DIFFUSION_SR, n_fft=n_fft)
    body_mask = ((freqs >= float(low_hz)) & (freqs <= float(high_hz))).astype(np.float32)[:, None]
    mixed_stft = base_stft * (1.0 - body_mask) + ((1.0 - float(alpha)) * base_stft + float(alpha) * donor_stft) * body_mask
    y = librosa.istft(mixed_stft, hop_length=hop, length=len(base))
    return _normalize(y)


def _linear_mix(base: np.ndarray, donor: np.ndarray, alpha: float) -> np.ndarray:
    base, donor = _match_length(base, donor)
    return _normalize((1.0 - float(alpha)) * base + float(alpha) * donor)


def _candidate_backings(retrieval: np.ndarray, pretrained: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "retrieval_raw": _normalize(retrieval),
        "pretrained_raw": _normalize(pretrained),
        "mix_020": _linear_mix(retrieval, pretrained, 0.20),
        "mix_035": _linear_mix(retrieval, pretrained, 0.35),
        "body_020": _body_inject(retrieval, pretrained, alpha=0.20),
        "body_035": _body_inject(retrieval, pretrained, alpha=0.35),
        "body_050": _body_inject(retrieval, pretrained, alpha=0.50),
    }


def _local_score(target_conf: float, target_margin: float, fullness: float, structure: float, warble: float) -> float:
    return float(0.50 * target_margin + 0.22 * target_conf + 0.20 * fullness + 0.12 * structure - 0.18 * warble)


def _row_key(song: str, target: str) -> str:
    return f"{song}__{target}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize a blend between retrieval and pretrained accompaniment packs.")
    ap.add_argument("--retrieval-manifest", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_retrieval_suite" / "suite_20260331_214339" / "selected_pack_final" / "manifest.csv")
    ap.add_argument("--pretrained-manifest", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_pretrained_encodec_aggressive_suite" / "suite_20260401_001640" / "combined_pack" / "manifest.csv")
    ap.add_argument("--judge-ckpt", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_retrieval_pretrained_blend")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / f"blendopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows = _read_manifest(Path(args.retrieval_manifest))
    pretrained_rows = _read_manifest(Path(args.pretrained_manifest))
    common_keys = sorted(set(retrieval_rows) & set(pretrained_rows))
    if not common_keys:
        raise RuntimeError("No overlapping song/target rows between retrieval and pretrained manifests")
    song_lookup = {_slug(Path(str(item["path"])).stem): item for item in picked_songs()}
    hybrid_cfg = HybridPushConfig()

    device = _device_from_arg(str(args.device))
    judge, _ = _load_or_train_judge(Path(args.judge_ckpt), Path(args.cache_dir), out_dir, device, 256)

    choice_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, str]] = []
    probs_for_sep: List[np.ndarray] = []
    per_target_winners: Dict[str, List[str]] = defaultdict(list)

    for song, target in common_keys:
        retrieval_wav, _ = sf.read(retrieval_rows[(song, target)]["accompaniment_wav"], dtype="float32")
        pretrained_wav, _ = sf.read(pretrained_rows[(song, target)]["accompaniment_wav"], dtype="float32")
        stems = _resolve_stems(hybrid_cfg, song_lookup[str(song)])
        source_acc, _ = sf.read(str(stems["accompaniment"]), dtype="float32")
        candidates = _candidate_backings(np.asarray(retrieval_wav, dtype=np.float32), np.asarray(pretrained_wav, dtype=np.float32))
        row_dir = out_dir / "rows" / song / target
        row_dir.mkdir(parents=True, exist_ok=True)
        best: Dict[str, Any] | None = None
        for label, backing in candidates.items():
            cand_dir = row_dir / label
            cand_dir.mkdir(parents=True, exist_ok=True)
            accomp_path = cand_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), backing, DIFFUSION_SR)
            hybrid_path = _mix_preserved_vocals(stems["vocals"], backing, cand_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = _judge_probs_for_audio(backing, judge, device, 256)
            target_idx = {"baroque_classical": 0, "cc0_other": 1, "hiphop_xtc": 2, "lofi_hh_lfbb": 3}[target]
            conf = float(probs[target_idx])
            margin = float(conf - float(np.max(np.delete(probs, target_idx))))
            metrics = _audio_metrics(source_acc, backing, DIFFUSION_SR)
            score = _local_score(conf, margin, float(metrics["fullness"]), float(metrics["structure"]), float(metrics["warble"]))
            payload = {
                "song": song,
                "target": target,
                "candidate": label,
                "local_score": score,
                "target_conf": conf,
                "target_margin": margin,
                "warble": float(metrics["warble"]),
                "fullness": float(metrics["fullness"]),
                "structure": float(metrics["structure"]),
                "judge_probs": probs.tolist(),
                "hybrid_wav": str(hybrid_path),
                "accompaniment_wav": str(accomp_path),
            }
            _write_json(cand_dir / "candidate_summary.json", payload)
            if best is None or payload["local_score"] > best["local_score"]:
                best = payload
        assert best is not None
        chosen_dir = row_dir / "selected"
        chosen_dir.mkdir(parents=True, exist_ok=True)
        chosen_backing = Path(str(best["accompaniment_wav"]))
        chosen_hybrid = Path(str(best["hybrid_wav"]))
        sf.write(str(chosen_dir / "accompaniment_generated.wav"), sf.read(str(chosen_backing), dtype="float32")[0], DIFFUSION_SR)
        sf.write(str(chosen_dir / "hybrid_longform_coherent.wav"), sf.read(str(chosen_hybrid), dtype="float32")[0], DIFFUSION_SR)
        choice_rows.append(best)
        probs_for_sep.append(np.asarray(best["judge_probs"], dtype=np.float32))
        per_target_winners[target].append(str(best["candidate"]))
        manifest_rows.append(
            {
                "target": target,
                "source_song": song,
                "source_target_dir": str(chosen_dir),
                "hybrid_wav": str(chosen_dir / "hybrid_longform_coherent.wav"),
                "accompaniment_wav": str(chosen_dir / "accompaniment_generated.wav"),
            }
        )

    sep_vals: List[float] = []
    by_song: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in choice_rows:
        by_song[str(row["song"])].append(np.asarray(row["judge_probs"], dtype=np.float32))
    for probs_list in by_song.values():
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                sep_vals.append(float(np.mean(np.abs(probs_list[i] - probs_list[j]))))
    mean_sep = float(np.mean(sep_vals)) if sep_vals else 0.0
    for row in choice_rows:
        row["separation"] = mean_sep
        row["overall"] = float(row["local_score"] + 0.20 * mean_sep)

    final_dir = out_dir / "selected_pack"
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    _write_json(final_dir / "summary.json", {
        "selected_pack_dir": str(final_dir),
        "mean_overall": float(np.mean([r["overall"] for r in choice_rows])),
        "mean_target_conf": float(np.mean([r["target_conf"] for r in choice_rows])),
        "mean_target_margin": float(np.mean([r["target_margin"] for r in choice_rows])),
        "mean_fullness": float(np.mean([r["fullness"] for r in choice_rows])),
        "mean_structure": float(np.mean([r["structure"] for r in choice_rows])),
        "mean_warble": float(np.mean([r["warble"] for r in choice_rows])),
        "mean_separation": mean_sep,
        "rows": choice_rows,
    })
    _write_json(final_dir / "winner_map.json", {target: winners for target, winners in per_target_winners.items()})
    (final_dir / "diagnosis_report.md").write_text(
        "\n".join(
            [
                "# Retrieval + Pretrained Blend Optimization",
                "",
                "- This pack chooses per song/target between retrieval-only, pretrained-only, and blended accompaniment variants.",
                f"- Mean target confidence: {float(np.mean([r['target_conf'] for r in choice_rows])):.4f}",
                f"- Mean target margin: {float(np.mean([r['target_margin'] for r in choice_rows])):.4f}",
                f"- Mean fullness: {float(np.mean([r['fullness'] for r in choice_rows])):.4f}",
                f"- Mean warble: {float(np.mean([r['warble'] for r in choice_rows])):.4f}",
                f"- Mean separation: {mean_sep:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(out_dir / "all_rows.json", {"rows": choice_rows})


if __name__ == "__main__":
    main()
