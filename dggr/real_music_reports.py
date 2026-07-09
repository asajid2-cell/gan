from __future__ import annotations

import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from .real_music_validation import _compact_metrics, _safe_corr, audio_metrics, load_audio_mono


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    return float(np.dot(x, y) / ((np.linalg.norm(x) + 1e-8) * (np.linalg.norm(y) + 1e-8)))


def _metric_vector(row: Dict[str, Any]) -> List[float]:
    keys = [
        "target_style_zdist",
        "source_style_zdist",
        "style_margin",
        "content_chroma_cos",
        "content_onset_corr",
        "content_rms_corr",
        "warble",
        "fullness",
        "hf_ratio",
        "lf_ratio",
    ]
    return [float(row.get(k, 0.0) or 0.0) for k in keys]


def write_final_pack(
    *,
    validation_pack_dir: Path,
    out_dir: Path,
    validation_report: Optional[Path] = None,
) -> Dict[str, Any]:
    pack_dir = Path(validation_pack_dir)
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation pack manifest missing: {manifest_path}")
    pack = _load_json(manifest_path)
    metrics_by_case: Dict[str, Dict[str, Any]] = {}
    if validation_report and Path(validation_report).exists():
        report = _load_json(Path(validation_report))
        metrics_by_case = {str(r["case_id"]): r for r in report.get("rows", [])}

    out_dir = Path(out_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for case in pack.get("rows", []):
        src = Path(str(case["source_audio"]))
        gen = Path(str(case["generated_wav"]))
        case_id = str(case["case_id"])
        target = str(case["target_genre"])
        dst = wav_dir / f"{case_id}__to__{target}.wav"
        if not gen.exists():
            raise FileNotFoundError(f"Generated WAV missing for {case_id}: {gen}")
        shutil.copy2(gen, dst)
        row = {
            "case_id": case_id,
            "source_audio": str(src),
            "source_genre": str(case.get("source_genre", "")),
            "target_genre": target,
            "generated_wav": str(dst),
            "source_track_id": str(case.get("track_id", "")),
        }
        row.update(metrics_by_case.get(case_id, {}))
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "final_pack_dir": str(out_dir),
        "manifest_csv": str(csv_path),
        "n_cases": int(len(rows)),
        "targets": sorted({str(r["target_genre"]) for r in rows}),
        "sources": sorted({str(r["source_genre"]) for r in rows}),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def genre_separation_report(*, validation_report: Path, out_path: Path) -> Dict[str, Any]:
    report = _load_json(Path(validation_report))
    rows = list(report.get("rows", []))
    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row.get("target_genre", ""))].append(row)

    target_stats: List[Dict[str, Any]] = []
    centroids: Dict[str, np.ndarray] = {}
    for target, target_rows in sorted(by_target.items()):
        vectors = np.asarray([_metric_vector(r) for r in target_rows], dtype=np.float32)
        if len(vectors):
            centroids[target] = np.mean(vectors, axis=0)
        target_stats.append(
            {
                "target_genre": target,
                "n_cases": int(len(target_rows)),
                "mean_target_style_cos": _mean(r.get("target_style_cos", 0.0) for r in target_rows),
                "mean_source_style_cos": _mean(r.get("source_style_cos", 0.0) for r in target_rows),
                "mean_style_margin": _mean(r.get("style_margin", 0.0) for r in target_rows),
                "mean_content_chroma_cos": _mean(r.get("content_chroma_cos", 0.0) for r in target_rows),
                "mean_warble": _mean(r.get("warble", 0.0) for r in target_rows),
                "mean_fullness": _mean(r.get("fullness", 0.0) for r in target_rows),
            }
        )

    pairwise: List[Dict[str, Any]] = []
    targets = sorted(centroids.keys())
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            sim = _cosine(centroids[a], centroids[b])
            pairwise.append({"target_a": a, "target_b": b, "centroid_cosine": float(sim), "separation": float(1.0 - sim)})

    summary = {
        "validation_report": str(validation_report),
        "n_cases": int(len(rows)),
        "n_targets": int(len(by_target)),
        "mean_style_margin": _mean(r.get("style_margin", 0.0) for r in rows),
        "mean_target_style_cos": _mean(r.get("target_style_cos", 0.0) for r in rows),
        "mean_content_chroma_cos": _mean(r.get("content_chroma_cos", 0.0) for r in rows),
        "mean_pairwise_target_separation": _mean(p["separation"] for p in pairwise),
        "targets": target_stats,
        "pairwise_target_separation": pairwise,
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(target_stats).to_csv(Path(out_path).with_suffix(".targets.csv"), index=False)
    pd.DataFrame(pairwise).to_csv(Path(out_path).with_suffix(".pairs.csv"), index=False)
    return summary


def novelty_and_listening_audit(
    *,
    validation_pack_dir: Path,
    validation_report: Path,
    out_path: Path,
    manual_notes_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    report = _load_json(Path(validation_report))
    metrics_by_case = {str(r["case_id"]): r for r in report.get("rows", [])}
    manual_by_case: Dict[str, Dict[str, Any]] = {}
    invalid_manual_rows: List[str] = []
    required_manual_fields = [
        "realism_pass",
        "source_identity_pass",
        "target_recognizable_pass",
        "artifact_free_pass",
        "novelty_pass",
        "baseline_preference",
    ]
    if manual_notes_csv and Path(manual_notes_csv).exists():
        with Path(manual_notes_csv).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                case_id = str(row.get("case_id", "")).strip()
                complete = str(row.get("review_complete", "")).strip().lower() in {"1", "true", "yes", "y"}
                has_required = all(str(row.get(field, "")).strip() for field in required_manual_fields)
                if case_id and complete and has_required:
                    manual_by_case[case_id] = dict(row)
                elif case_id:
                    invalid_manual_rows.append(case_id)

    rows: List[Dict[str, Any]] = []
    for case in pack.get("rows", []):
        case_id = str(case["case_id"])
        gen = Path(str(case["generated_wav"]))
        source = Path(str(case["source_audio"]))
        if not gen.exists():
            raise FileNotFoundError(gen)
        gen_audio = load_audio_mono(gen)
        src_audio = load_audio_mono(source, seconds=float(case.get("seconds", 0.0)))
        gen_m = audio_metrics(gen_audio)
        src_m = audio_metrics(src_audio)
        novelty_proxy = float(
            abs(float(gen_m["centroid_mean"]) - float(src_m["centroid_mean"])) / 5000.0
            + abs(float(gen_m["onset_mean"]) - float(src_m["onset_mean"]))
            + abs(float(gen_m["high_ratio"]) - float(src_m["high_ratio"]))
            + abs(float(gen_m["low_ratio"]) - float(src_m["low_ratio"]))
        )
        row = {
            "case_id": case_id,
            "source_genre": str(case.get("source_genre", "")),
            "target_genre": str(case.get("target_genre", "")),
            "generated_wav": str(gen),
            "novelty_proxy": float(novelty_proxy),
            "structure_rms_corr": float(_safe_corr(np.asarray(src_m["rms_env"]), np.asarray(gen_m["rms_env"]))),
            "style_margin": float(metrics_by_case.get(case_id, {}).get("style_margin", 0.0) or 0.0),
            "content_chroma_cos": float(metrics_by_case.get(case_id, {}).get("content_chroma_cos", 0.0) or 0.0),
            "manual_review_status": "PRESENT" if case_id in manual_by_case else "MISSING",
        }
        row.update({f"manual_{k}": v for k, v in manual_by_case.get(case_id, {}).items() if k != "case_id"})
        rows.append(row)

    summary = {
        "validation_pack_dir": str(validation_pack_dir),
        "validation_report": str(validation_report),
        "manual_notes_csv": str(manual_notes_csv) if manual_notes_csv else "",
        "n_cases": int(len(rows)),
        "manual_reviews_present": int(sum(1 for r in rows if r["manual_review_status"] == "PRESENT")),
        "invalid_manual_rows": int(len(invalid_manual_rows)),
        "manual_required_fields": required_manual_fields,
        "manual_review_required": bool(len(rows) == 0 or any(r["manual_review_status"] != "PRESENT" for r in rows)),
        "mean_novelty_proxy": _mean(r["novelty_proxy"] for r in rows),
        "mean_style_margin": _mean(r["style_margin"] for r in rows),
        "mean_content_chroma_cos": _mean(r["content_chroma_cos"] for r in rows),
        "rows": rows,
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    return summary


def manual_review_packet(
    *,
    validation_pack_dir: Path,
    validation_report: Path,
    baseline_pack_dir: Path,
    baseline_validation_report: Path,
    out_dir: Path,
    title: str = "Real-Music Transfer Manual Review",
) -> Dict[str, Any]:
    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    new_report = _load_json(Path(validation_report)) if Path(validation_report).exists() else {}
    baseline_pack = _load_json(Path(baseline_pack_dir) / "manifest.json") if (Path(baseline_pack_dir) / "manifest.json").exists() else {}
    baseline_report = _load_json(Path(baseline_validation_report)) if Path(baseline_validation_report).exists() else {}
    new_metrics = {str(r.get("case_id", "")): r for r in new_report.get("rows", [])}
    base_metrics = {str(r.get("case_id", "")): r for r in baseline_report.get("rows", [])}
    base_rows = {str(r.get("case_id", "")): r for r in baseline_pack.get("rows", [])}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for case in pack.get("rows", []):
        case_id = str(case.get("case_id", ""))
        new_m = new_metrics.get(case_id, {})
        base_m = base_metrics.get(case_id, {})
        base_case = base_rows.get(case_id, {})
        row = {
            "case_id": case_id,
            "reviewer": "",
            "review_complete": "0",
            "realism_pass": "",
            "source_identity_pass": "",
            "target_recognizable_pass": "",
            "artifact_free_pass": "",
            "novelty_pass": "",
            "baseline_preference": "",
            "notes": "",
            "source_audio": str(case.get("source_audio", "")),
            "new_wav": str(case.get("generated_wav", "")),
            "baseline_wav": str(base_case.get("generated_wav", "")),
            "source_genre": str(case.get("source_genre", "")),
            "target_genre": str(case.get("target_genre", "")),
            "codec_target_genre": str(base_case.get("codec_target_genre", "")),
            "new_style_margin": new_m.get("style_margin", ""),
            "baseline_style_margin": base_m.get("style_margin", ""),
            "new_content_chroma": new_m.get("content_chroma_cos", ""),
            "baseline_content_chroma": base_m.get("content_chroma_cos", ""),
            "new_warble": new_m.get("warble", ""),
            "baseline_warble": base_m.get("warble", ""),
        }
        rows.append(row)

    template_csv = out_dir / "manual_notes_template.csv"
    pd.DataFrame(rows).to_csv(template_csv, index=False)

    def _priority(row: Dict[str, Any]) -> float:
        vals = []
        for key in ("new_style_margin", "new_content_chroma", "new_warble"):
            try:
                vals.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                vals.append(0.0)
        style_margin, content_chroma, warble = vals
        return float((1.0 - content_chroma) + max(0.0, 0.20 - style_margin) + max(0.0, warble - 0.20))

    priority_rows = sorted(rows, key=_priority, reverse=True)[: min(32, len(rows))]
    priority_csv = out_dir / "priority_cases.csv"
    pd.DataFrame(priority_rows).to_csv(priority_csv, index=False)

    html_rows = []
    for row in rows:
        def esc(key: str) -> str:
            return html.escape(str(row.get(key, "")), quote=True)

        html_rows.append(
            "<tr>"
            f"<td>{esc('case_id')}</td>"
            f"<td>{esc('source_genre')}<br>to<br>{esc('target_genre')}</td>"
            f"<td><audio controls preload=\"none\" src=\"{esc('source_audio')}\"></audio></td>"
            f"<td><audio controls preload=\"none\" src=\"{esc('new_wav')}\"></audio></td>"
            f"<td><audio controls preload=\"none\" src=\"{esc('baseline_wav')}\"></audio></td>"
            f"<td>new margin {esc('new_style_margin')}<br>base margin {esc('baseline_style_margin')}<br>new chroma {esc('new_content_chroma')}<br>base chroma {esc('baseline_content_chroma')}</td>"
            "</tr>"
        )
    index_html = out_dir / "index.html"
    index_html.write_text(
        "\n".join(
            [
                "<!doctype html><meta charset=\"utf-8\">",
                f"<title>{html.escape(title)}</title>",
                "<style>body{font-family:Arial,sans-serif;margin:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}audio{width:260px}</style>",
                f"<h1>{html.escape(title)}</h1>",
                "<p>Fill manual_notes_template.csv. Required pass fields: realism_pass, source_identity_pass, target_recognizable_pass, artifact_free_pass, novelty_pass, baseline_preference, with review_complete=1.</p>",
                "<table><thead><tr><th>Case</th><th>Transfer</th><th>Source</th><th>New</th><th>Baseline</th><th>Metrics</th></tr></thead><tbody>",
                *html_rows,
                "</tbody></table>",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "out_dir": str(out_dir),
        "validation_pack_dir": str(validation_pack_dir),
        "validation_report": str(validation_report),
        "baseline_pack_dir": str(baseline_pack_dir),
        "baseline_validation_report": str(baseline_validation_report),
        "n_cases": int(len(rows)),
        "index_html": str(index_html),
        "manual_notes_template_csv": str(template_csv),
        "priority_cases_csv": str(priority_csv),
        "required_manual_fields": [
            "review_complete",
            "realism_pass",
            "source_identity_pass",
            "target_recognizable_pass",
            "artifact_free_pass",
            "novelty_pass",
            "baseline_preference",
        ],
    }
    _write_json(out_dir / "README.json", summary)
    return summary


def baseline_compare_report(
    *,
    new_validation_report: Path,
    baseline_validation_report: Optional[Path],
    out_path: Path,
) -> Dict[str, Any]:
    new_report = _load_json(Path(new_validation_report))
    base_report = _load_json(Path(baseline_validation_report)) if baseline_validation_report and Path(baseline_validation_report).exists() else None
    keys = [
        "mean_target_style_cos",
        "mean_style_margin",
        "mean_content_chroma_cos",
        "mean_content_onset_corr",
        "mean_content_rms_corr",
        "mean_warble",
        "mean_fullness",
    ]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        new_val = float(new_report.get(key, 0.0) or 0.0)
        base_val = float(base_report.get(key, 0.0) or 0.0) if base_report else None
        rows.append(
            {
                "metric": key,
                "new_model": new_val,
                "baseline": base_val,
                "delta_new_minus_baseline": None if base_val is None else float(new_val - base_val),
            }
        )
    summary = {
        "new_validation_report": str(new_validation_report),
        "baseline_validation_report": str(baseline_validation_report) if baseline_validation_report else "",
        "baseline_available": bool(base_report is not None),
        "rows": rows,
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    return summary


def realism_distribution_report(
    *,
    validation_pack_dir: Path,
    out_path: Path,
    reference_profiles: Path,
) -> Dict[str, Any]:
    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    profiles = _load_json(Path(reference_profiles)).get("profiles", {}) if Path(reference_profiles).exists() else {}
    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for case in pack.get("rows", []):
        target = str(case.get("target_genre", ""))
        gen_y = load_audio_mono(Path(str(case["generated_wav"])))
        metrics = audio_metrics(gen_y)
        compact = _compact_metrics(metrics)
        row = {
            "case_id": str(case.get("case_id", "")),
            "target_genre": target,
            "generated_wav": str(case.get("generated_wav", "")),
            "warble": float(metrics["warble"]),
            "fullness": float(metrics["fullness"]),
            "dynamic_range_db": float(metrics["dynamic_range_db"]),
            "hf_ratio": float(metrics["high_ratio"]),
            "lf_ratio": float(metrics["low_ratio"]),
            "_compact": compact,
        }
        rows.append(row)
        by_target[target].append(row)

    target_rows: List[Dict[str, Any]] = []
    for target, target_cases in sorted(by_target.items()):
        profile = profiles.get(str(target), {})
        ref_mean = np.asarray(profile.get("mean", []), dtype=np.float32)
        ref_std = np.maximum(np.asarray(profile.get("std", []), dtype=np.float32), 1e-3)
        gen_feats = np.stack([np.asarray(r["_compact"], dtype=np.float32) for r in target_cases])
        gen_mean = gen_feats.mean(axis=0)
        gen_std = np.maximum(gen_feats.std(axis=0), 1e-6)
        if len(ref_mean) == gen_feats.shape[1] and len(ref_std) == gen_feats.shape[1]:
            diag_frechet = float(np.sum((gen_mean - ref_mean) ** 2 + gen_std**2 + ref_std**2 - 2.0 * gen_std * ref_std))
            mean_abs_z = float(np.mean(np.abs((gen_feats - ref_mean[None, :]) / ref_std[None, :])))
            n_ref = int(profile.get("n", 0) or 0)
        else:
            diag_frechet = float("nan")
            mean_abs_z = float("nan")
            n_ref = 0
        target_rows.append(
            {
                "target_genre": target,
                "n_generated": int(len(target_cases)),
                "n_reference": n_ref,
                "diag_frechet_audio_feature_distance": diag_frechet,
                "mean_abs_reference_z_distance": mean_abs_z,
            }
        )

    clean_rows = [{k: v for k, v in r.items() if k != "_compact"} for r in rows]
    summary = {
        "validation_pack": str(Path(validation_pack_dir) / "manifest.json"),
        "reference_profiles": str(reference_profiles),
        "n_cases": int(len(rows)),
        "n_targets": int(len(by_target)),
        "metric_note": "Diagonal Frechet-style distance over compact audio features from real_music_validation._compact_metrics; lower is closer to the real target-family audio-feature distribution. This is a FAD substitute, not a human listening result.",
        "mean_diag_frechet_audio_feature_distance": _mean(r["diag_frechet_audio_feature_distance"] for r in target_rows),
        "mean_abs_reference_z_distance": _mean(r["mean_abs_reference_z_distance"] for r in target_rows),
        "mean_warble": _mean(r["warble"] for r in clean_rows),
        "mean_fullness": _mean(r["fullness"] for r in clean_rows),
        "targets": target_rows,
        "rows": clean_rows,
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(target_rows).to_csv(Path(out_path).with_suffix(".targets.csv"), index=False)
    pd.DataFrame(clean_rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    return summary


def mert_realism_report(
    *,
    validation_pack_dir: Path,
    out_path: Path,
    reference_profiles: Path,
    model_name: str = "m-a-p/MERT-v1-95M",
    seconds: float = 12.0,
    refs_per_target: int = 6,
    max_cases: int = 0,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if str(device_arg).lower() == "auto" and torch.cuda.is_available() else ("cpu" if str(device_arg).lower() == "auto" else str(device_arg)))
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True).to(device)
    model.eval()
    sample_rate = int(getattr(processor, "sampling_rate", 24000) or 24000)

    def _embed(path: Path) -> np.ndarray:
        y, _sr = librosa.load(str(path), sr=sample_rate, mono=True, duration=float(seconds), dtype=np.float32)
        if len(y) == 0:
            y = np.zeros((sample_rate,), dtype=np.float32)
        inputs = processor(y, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and hasattr(out, "hidden_states") and out.hidden_states:
            hidden = out.hidden_states[-1]
        if hidden is None:
            raise RuntimeError("MERT model did not return hidden states")
        emb = hidden.mean(dim=1).detach().cpu().numpy()[0].astype(np.float32)
        return emb

    def _diag_fad(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        mu_a = a.mean(axis=0)
        mu_b = b.mean(axis=0)
        std_a = np.maximum(a.std(axis=0), 1e-6)
        std_b = np.maximum(b.std(axis=0), 1e-6)
        return float(np.sum((mu_a - mu_b) ** 2 + std_a**2 + std_b**2 - 2.0 * std_a * std_b))

    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    profiles = _load_json(Path(reference_profiles)).get("profiles", {})
    pack_rows = list(pack.get("rows", []))
    if int(max_cases) > 0:
        pack_rows = pack_rows[: int(max_cases)]

    generated_by_target: Dict[str, List[np.ndarray]] = defaultdict(list)
    generated_rows: List[Dict[str, Any]] = []
    for case in pack_rows:
        target = str(case.get("target_genre", ""))
        wav = Path(str(case["generated_wav"]))
        emb = _embed(wav)
        generated_by_target[target].append(emb)
        generated_rows.append({"case_id": str(case.get("case_id", "")), "target_genre": target, "generated_wav": str(wav)})

    reference_by_target: Dict[str, List[np.ndarray]] = defaultdict(list)
    reference_rows: List[Dict[str, Any]] = []
    for target, profile in sorted(profiles.items()):
        if target not in generated_by_target:
            continue
        for ref_path in list(profile.get("examples", []))[: max(1, int(refs_per_target))]:
            path = Path(str(ref_path))
            if not path.exists():
                continue
            emb = _embed(path)
            reference_by_target[str(target)].append(emb)
            reference_rows.append({"target_genre": str(target), "reference_audio": str(path)})

    target_rows: List[Dict[str, Any]] = []
    for target, gen_embs in sorted(generated_by_target.items()):
        ref_embs = reference_by_target.get(target, [])
        if gen_embs and ref_embs:
            gen_arr = np.stack(gen_embs)
            ref_arr = np.stack(ref_embs)
            fad = _diag_fad(gen_arr, ref_arr)
            centroid_cos = _cosine(gen_arr.mean(axis=0), ref_arr.mean(axis=0))
        else:
            fad = float("nan")
            centroid_cos = float("nan")
        target_rows.append(
            {
                "target_genre": target,
                "n_generated": int(len(gen_embs)),
                "n_reference": int(len(ref_embs)),
                "mert_diag_fad": fad,
                "mert_centroid_cosine": centroid_cos,
            }
        )

    summary = {
        "validation_pack": str(Path(validation_pack_dir) / "manifest.json"),
        "reference_profiles": str(reference_profiles),
        "model_name": str(model_name),
        "seconds": float(seconds),
        "refs_per_target": int(refs_per_target),
        "device": str(device),
        "n_cases": int(len(generated_rows)),
        "n_targets": int(len(generated_by_target)),
        "n_reference": int(len(reference_rows)),
        "mean_mert_diag_fad": _mean(r["mert_diag_fad"] for r in target_rows),
        "mean_mert_centroid_cosine": _mean(r["mert_centroid_cosine"] for r in target_rows),
        "targets": target_rows,
        "rows": generated_rows,
        "references": reference_rows,
        "note": "MERT embedding diagonal-FAD report using a locally cached model; this is stronger than compact handcrafted features but still does not replace human listening.",
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(target_rows).to_csv(Path(out_path).with_suffix(".targets.csv"), index=False)
    pd.DataFrame(generated_rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    pd.DataFrame(reference_rows).to_csv(Path(out_path).with_suffix(".references.csv"), index=False)
    return summary


def longform_coherence_report(
    *,
    generated_wav: Path,
    source_audio: Path,
    out_path: Path,
    source_genre: str = "",
    target_genre: str = "",
    reference_profiles: Optional[Path] = None,
    seconds: float = 0.0,
    expected_seconds: float = 0.0,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.5,
    boundary_window_seconds: float = 0.25,
) -> Dict[str, Any]:
    gen_y = load_audio_mono(Path(generated_wav))
    src_y = load_audio_mono(Path(source_audio), seconds=float(seconds) if float(seconds) > 0 else float(expected_seconds))
    gen_m = audio_metrics(gen_y)
    src_m = audio_metrics(src_y)

    sr = 22050
    hop_seconds = max(1e-3, float(chunk_seconds) - float(overlap_seconds))
    boundary_count = int(math.floor(max(0.0, float(len(gen_y) / sr) - float(overlap_seconds)) / hop_seconds))
    window = max(1, int(round(float(boundary_window_seconds) * sr)))
    boundaries: List[Dict[str, Any]] = []
    for i in range(1, boundary_count + 1):
        center = int(round(i * hop_seconds * sr))
        if center - window < 0 or center + window > len(gen_y):
            continue
        left = gen_y[center - window : center]
        right = gen_y[center : center + window]
        left_rms = float(np.sqrt(np.mean(np.square(left), dtype=np.float64)) + 1e-8)
        right_rms = float(np.sqrt(np.mean(np.square(right), dtype=np.float64)) + 1e-8)
        jump = float(abs(right_rms - left_rms) / max(left_rms, right_rms, 1e-8))
        boundaries.append({"time_sec": float(center / sr), "rms_jump": jump, "left_rms": left_rms, "right_rms": right_rms})

    style_metrics: Dict[str, float] = {}
    if reference_profiles and Path(reference_profiles).exists() and source_genre and target_genre:
        profiles = _load_json(Path(reference_profiles)).get("profiles", {})
        target_profile = profiles.get(str(target_genre), {})
        source_profile = profiles.get(str(source_genre), {})
        target_mean = np.asarray(target_profile.get("mean", []), dtype=np.float32)
        source_mean = np.asarray(source_profile.get("mean", []), dtype=np.float32)
        target_std = np.maximum(np.asarray(target_profile.get("std", []), dtype=np.float32), 1e-3)
        source_std = np.maximum(np.asarray(source_profile.get("std", []), dtype=np.float32), 1e-3)
        gen_vec = _compact_metrics(gen_m)
        if len(target_mean) == len(gen_vec) and len(target_std) == len(gen_vec):
            style_metrics["target_style_zdist"] = float(np.mean(np.abs((gen_vec - target_mean) / target_std)))
        if len(source_mean) == len(gen_vec) and len(source_std) == len(gen_vec):
            style_metrics["source_style_zdist"] = float(np.mean(np.abs((gen_vec - source_mean) / source_std)))
        if "source_style_zdist" in style_metrics and "target_style_zdist" in style_metrics:
            style_metrics["style_margin"] = float(style_metrics["source_style_zdist"] - style_metrics["target_style_zdist"])

    summary = {
        "generated_wav": str(generated_wav),
        "source_audio": str(source_audio),
        "source_genre": str(source_genre),
        "target_genre": str(target_genre),
        "reference_profiles": str(reference_profiles) if reference_profiles else "",
        "duration_sec": float(len(gen_y) / sr),
        "expected_seconds": float(expected_seconds),
        "chunk_seconds": float(chunk_seconds),
        "overlap_seconds": float(overlap_seconds),
        "boundary_window_seconds": float(boundary_window_seconds),
        "boundary_count": int(len(boundaries)),
        "mean_boundary_rms_jump": _mean(b["rms_jump"] for b in boundaries),
        "max_boundary_rms_jump": float(max([b["rms_jump"] for b in boundaries], default=0.0)),
        "content_chroma_cos": _cosine(src_m["chroma_mean"], gen_m["chroma_mean"]),
        "content_onset_corr": float(_safe_corr(np.asarray(src_m["onset_env"]), np.asarray(gen_m["onset_env"]))),
        "content_rms_corr": float(_safe_corr(np.asarray(src_m["rms_env"]), np.asarray(gen_m["rms_env"]))),
        "warble": float(gen_m["warble"]),
        "fullness": float(gen_m["fullness"]),
        "dynamic_range_db": float(gen_m["dynamic_range_db"]),
        "boundaries": boundaries,
    }
    summary.update(style_metrics)
    _write_json(Path(out_path), summary)
    pd.DataFrame(boundaries).to_csv(Path(out_path).with_suffix(".boundaries.csv"), index=False)
    return summary


def content_structure_report(*, validation_pack_dir: Path, out_path: Path, seconds: float = 0.0) -> Dict[str, Any]:
    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    rows: List[Dict[str, Any]] = []

    def _frame_chroma(y: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_stft(y=np.asarray(y, dtype=np.float32), sr=22050, n_fft=2048, hop_length=512)
        norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
        return (chroma / norm).astype(np.float32)

    def _segment_chroma(chroma: np.ndarray, n_segments: int = 8) -> np.ndarray:
        if chroma.shape[1] == 0:
            return np.zeros((12, n_segments), dtype=np.float32)
        edges = np.linspace(0, chroma.shape[1], int(n_segments) + 1).astype(int)
        segs = []
        for i in range(int(n_segments)):
            a, b = int(edges[i]), int(edges[i + 1])
            part = chroma[:, a:max(a + 1, b)]
            seg = part.mean(axis=1)
            segs.append(seg / (np.linalg.norm(seg) + 1e-8))
        return np.stack(segs, axis=1).astype(np.float32)

    def _aligned_frame_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = min(a.shape[1], b.shape[1])
        if n == 0:
            return np.zeros((1,), dtype=np.float32)
        return np.sum(a[:, :n] * b[:, :n], axis=0).astype(np.float32)

    def _chroma_dtw_cost(a: np.ndarray, b: np.ndarray, max_frames: int = 900) -> float:
        if a.shape[1] == 0 or b.shape[1] == 0:
            return 1.0
        step_a = max(1, int(math.ceil(a.shape[1] / float(max_frames))))
        step_b = max(1, int(math.ceil(b.shape[1] / float(max_frames))))
        aa = a[:, ::step_a].T
        bb = b[:, ::step_b].T
        dist = 1.0 - np.clip(aa @ bb.T, -1.0, 1.0)
        cost, _wp = librosa.sequence.dtw(C=dist.astype(np.float32), backtrack=True)
        denom = max(1, aa.shape[0] + bb.shape[0])
        return float(cost[-1, -1] / float(denom))

    for case in pack.get("rows", []):
        src_y = load_audio_mono(Path(str(case["source_audio"])), seconds=float(seconds or case.get("seconds", 0.0)))
        gen_y = load_audio_mono(Path(str(case["generated_wav"])), seconds=float(seconds or case.get("seconds", 0.0)))
        src_chroma = _frame_chroma(src_y)
        gen_chroma = _frame_chroma(gen_y)
        frame_cos = _aligned_frame_cos(src_chroma, gen_chroma)
        src_seg = _segment_chroma(src_chroma)
        gen_seg = _segment_chroma(gen_chroma)
        seg_cos = _aligned_frame_cos(src_seg, gen_seg)
        src_m = audio_metrics(src_y)
        gen_m = audio_metrics(gen_y)
        row = {
            "case_id": str(case.get("case_id", "")),
            "source_genre": str(case.get("source_genre", "")),
            "target_genre": str(case.get("target_genre", "")),
            "source_audio": str(case.get("source_audio", "")),
            "generated_wav": str(case.get("generated_wav", "")),
            "frame_chroma_cos_mean": _mean(frame_cos),
            "frame_chroma_cos_p10": float(np.percentile(frame_cos, 10)),
            "segment_chroma_cos_mean": _mean(seg_cos),
            "segment_chroma_cos_p10": float(np.percentile(seg_cos, 10)),
            "chroma_dtw_cosine_cost": _chroma_dtw_cost(src_chroma, gen_chroma),
            "onset_env_corr": float(_safe_corr(np.asarray(src_m["onset_env"]), np.asarray(gen_m["onset_env"]))),
            "rms_env_corr": float(_safe_corr(np.asarray(src_m["rms_env"]), np.asarray(gen_m["rms_env"]))),
        }
        rows.append(row)

    summary = {
        "validation_pack_dir": str(validation_pack_dir),
        "n_cases": int(len(rows)),
        "mean_frame_chroma_cos": _mean(r["frame_chroma_cos_mean"] for r in rows),
        "mean_frame_chroma_cos_p10": _mean(r["frame_chroma_cos_p10"] for r in rows),
        "mean_segment_chroma_cos": _mean(r["segment_chroma_cos_mean"] for r in rows),
        "mean_segment_chroma_cos_p10": _mean(r["segment_chroma_cos_p10"] for r in rows),
        "mean_chroma_dtw_cosine_cost": _mean(r["chroma_dtw_cosine_cost"] for r in rows),
        "mean_onset_env_corr": _mean(r["onset_env_corr"] for r in rows),
        "mean_rms_env_corr": _mean(r["rms_env_corr"] for r in rows),
        "rows": rows,
        "note": "Automated chroma/segment/DTW and envelope structure proxy; this supports content validation but does not replace human song-identity listening.",
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    return summary


def musical_element_shift_report(
    *,
    validation_pack_dir: Path,
    out_path: Path,
    reference_profiles: Optional[Path] = None,
) -> Dict[str, Any]:
    pack = _load_json(Path(validation_pack_dir) / "manifest.json")
    profiles = _load_json(Path(reference_profiles)).get("profiles", {}) if reference_profiles and Path(reference_profiles).exists() else {}
    element_slices = {
        "tempo": (0, 1),
        "rms": (1, 2),
        "spectral_centroid": (2, 3),
        "spectral_flatness": (3, 4),
        "zero_crossing_rate": (4, 5),
        "low_mid_high_balance": (5, 8),
        "dynamic_range": (8, 9),
        "onset_strength": (9, 10),
        "chroma_profile": (10, 22),
    }
    rows: List[Dict[str, Any]] = []
    element_totals: Dict[str, List[float]] = {name: [] for name in element_slices}
    target_z_by_element: Dict[str, List[float]] = {name: [] for name in element_slices}
    novelty_by_element: Dict[str, List[float]] = {name: [] for name in element_slices}

    for case in pack.get("rows", []):
        gen_y = load_audio_mono(Path(str(case["generated_wav"])))
        src_y = load_audio_mono(Path(str(case["source_audio"])), seconds=float(case.get("seconds", 0.0)))
        gen_vec = _compact_metrics(audio_metrics(gen_y))
        src_vec = _compact_metrics(audio_metrics(src_y))
        target = str(case.get("target_genre", ""))
        target_profile = profiles.get(target, {})
        target_mean = np.asarray(target_profile.get("mean", []), dtype=np.float32)
        target_std = np.maximum(np.asarray(target_profile.get("std", []), dtype=np.float32), 1e-3)

        element_result: Dict[str, Any] = {}
        moved_count = 0
        available_count = 0
        for name, (start, end) in element_slices.items():
            gen_part = gen_vec[start:end]
            src_part = src_vec[start:end]
            if len(target_mean) >= end and len(target_std) >= end:
                target_part = target_mean[start:end]
                std_part = target_std[start:end]
                gen_target_z = float(np.mean(np.abs((gen_part - target_part) / std_part)))
                src_target_z = float(np.mean(np.abs((src_part - target_part) / std_part)))
                moved = bool(gen_target_z < src_target_z)
                element_result[f"{name}_moved_toward_target"] = moved
                element_result[f"{name}_source_target_z"] = src_target_z
                element_result[f"{name}_generated_target_z"] = gen_target_z
                element_result[f"{name}_target_z_delta_generated_minus_source"] = float(gen_target_z - src_target_z)
                element_totals[name].append(1.0 if moved else 0.0)
                target_z_by_element[name].append(gen_target_z)
                moved_count += int(moved)
                available_count += 1
            novelty = float(np.mean(np.abs(gen_part - src_part)))
            element_result[f"{name}_generated_minus_source_abs"] = novelty
            novelty_by_element[name].append(novelty)

        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "source_genre": str(case.get("source_genre", "")),
                "target_genre": target,
                "source_audio": str(case.get("source_audio", "")),
                "generated_wav": str(case.get("generated_wav", "")),
                "elements_moved_toward_target": int(moved_count),
                "elements_available": int(available_count),
                "element_target_movement_fraction": float(moved_count / max(1, available_count)),
                **element_result,
            }
        )

    element_summary = []
    for name in element_slices:
        element_summary.append(
            {
                "element": name,
                "target_movement_fraction": _mean(element_totals[name]),
                "mean_generated_target_z": _mean(target_z_by_element[name]),
                "mean_generated_minus_source_abs": _mean(novelty_by_element[name]),
            }
        )
    summary = {
        "validation_pack_dir": str(validation_pack_dir),
        "reference_profiles": str(reference_profiles) if reference_profiles else "",
        "n_cases": int(len(rows)),
        "mean_element_target_movement_fraction": _mean(r["element_target_movement_fraction"] for r in rows),
        "mean_elements_moved_toward_target": _mean(r["elements_moved_toward_target"] for r in rows),
        "elements": element_summary,
        "rows": rows,
        "note": "Automated audio-feature element movement report; this supports triage but does not replace human recognition of instrumentation, groove, or target genre.",
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    pd.DataFrame(element_summary).to_csv(Path(out_path).with_suffix(".elements.csv"), index=False)
    return summary


def lab1_bottleneck_audit_report(
    *,
    cache_dir: Path,
    out_path: Path,
    sample_size: int = 40000,
    retrieval_sample_size: int = 5000,
    seed: int = 328,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    rng = np.random.default_rng(int(seed))
    genre_idx = np.load(cache_dir / "diff_genre_idx.npy", mmap_mode="r")
    z_style = np.load(cache_dir / "diff_z_style.npy", mmap_mode="r")
    z_content = np.load(cache_dir / "diff_z_content.npy", mmap_mode="r")
    index_df = pd.read_csv(cache_dir / "diff_index.csv", usecols=["track_id", "genre"])
    n_total = int(len(genre_idx))
    n_sample = min(int(sample_size), n_total)
    sample_idx = rng.choice(n_total, size=n_sample, replace=False) if n_sample < n_total else np.arange(n_total)
    sample_idx.sort()

    y = np.asarray(genre_idx[sample_idx], dtype=np.int64)
    z_style_sample = np.asarray(z_style[sample_idx], dtype=np.float32)
    z_content_sample = np.asarray(z_content[sample_idx], dtype=np.float32)

    def _nearest_centroid_report(x: np.ndarray, labels: np.ndarray, train_frac: float = 0.7) -> Dict[str, Any]:
        train_mask = np.zeros(len(labels), dtype=bool)
        for g in np.unique(labels):
            pos = np.flatnonzero(labels == g)
            rng.shuffle(pos)
            train_n = max(1, int(round(len(pos) * float(train_frac))))
            train_mask[pos[:train_n]] = True
        test_mask = ~train_mask
        if not np.any(test_mask):
            test_mask[:] = True
        x_train = x[train_mask]
        y_train = labels[train_mask]
        x_test = x[test_mask]
        y_test = labels[test_mask]
        labels_unique = np.unique(labels)
        centroids = np.stack([x_train[y_train == g].mean(axis=0) for g in labels_unique]).astype(np.float32)
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        x_norm = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-8)
        pred = labels_unique[np.argmax(x_norm @ centroids.T, axis=1)]
        per_class = []
        for g in labels_unique:
            m = y_test == g
            if np.any(m):
                per_class.append(float(np.mean(pred[m] == y_test[m])))
        counts = np.bincount(labels, minlength=int(labels_unique.max()) + 1)
        random_majority = float(np.max(counts) / max(1, len(labels)))
        return {
            "accuracy": float(np.mean(pred == y_test)),
            "balanced_accuracy": _mean(per_class),
            "random_majority_baseline": random_majority,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
        }

    def _centroid_separation_report(x: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        labels_unique = np.unique(labels)
        centroids = np.stack([x[labels == g].mean(axis=0) for g in labels_unique]).astype(np.float32)
        within = []
        for i, g in enumerate(labels_unique):
            diffs = x[labels == g] - centroids[i]
            within.append(float(np.mean(np.linalg.norm(diffs, axis=1))))
        between = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                between.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        return {
            "mean_within_label_distance": _mean(within),
            "mean_between_label_centroid_distance": _mean(between),
            "between_to_within_ratio": float(_mean(between) / max(1e-8, _mean(within))),
        }

    def _same_track_retrieval_report(x: np.ndarray, all_indices: np.ndarray) -> Dict[str, Any]:
        n_retrieval = min(int(retrieval_sample_size), len(all_indices))
        if n_retrieval <= 1:
            return {"n": int(n_retrieval), "top1_same_track": 0.0, "random_same_track_baseline": 0.0}
        chosen = rng.choice(len(all_indices), size=n_retrieval, replace=False)
        source_idx = all_indices[chosen]
        tracks = index_df.iloc[source_idx]["track_id"].astype(str).to_numpy()
        emb = np.asarray(x[chosen], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sims = emb @ emb.T
        np.fill_diagonal(sims, -np.inf)
        nn = np.argmax(sims, axis=1)
        counts = pd.Series(tracks).value_counts().to_numpy(dtype=np.float64)
        random_same = float(np.sum(counts * (counts - 1.0)) / max(1.0, float(n_retrieval * (n_retrieval - 1))))
        return {
            "n": int(n_retrieval),
            "top1_same_track": float(np.mean(tracks[nn] == tracks)),
            "random_same_track_baseline": random_same,
        }

    style_probe = _nearest_centroid_report(z_style_sample, y)
    content_style_leakage = _nearest_centroid_report(z_content_sample, y)
    style_separation = _centroid_separation_report(z_style_sample, y)
    content_track_retrieval = _same_track_retrieval_report(z_content_sample, sample_idx)
    style_track_retrieval = _same_track_retrieval_report(z_style_sample, sample_idx)

    direct_feature_files = {
        "mel": cache_dir / "diff_mel.npy",
        "chroma": cache_dir / "diff_chroma.npy",
        "onset": cache_dir / "diff_onset.npy",
        "beat": cache_dir / "diff_beat.npy",
        "genre_idx": cache_dir / "diff_genre_idx.npy",
    }
    latent_files = {
        "z_content": cache_dir / "diff_z_content.npy",
        "z_style": cache_dir / "diff_z_style.npy",
    }
    direct_features_present = all(p.exists() for p in direct_feature_files.values())
    latents_present = all(p.exists() for p in latent_files.values())
    weak_style_latents = bool(style_probe["balanced_accuracy"] < 0.25 or style_separation["between_to_within_ratio"] < 0.2)
    content_identity_signal = bool(
        content_track_retrieval["top1_same_track"] > max(0.05, 3.0 * content_track_retrieval["random_same_track_baseline"])
    )
    production_bypasses_lab1_latents = bool(direct_features_present)
    bottleneck_risk = "LOW_FOR_CURRENT_REAL_MUSIC_MODEL" if production_bypasses_lab1_latents else "HIGH"
    if not production_bypasses_lab1_latents:
        bottleneck_risk = "HIGH"
    elif weak_style_latents:
        bottleneck_risk = "LOW_FOR_CURRENT_REAL_MUSIC_MODEL_BUT_HIGH_IF_REUSED_AS_STYLE_BOTTLENECK"

    summary = {
        "cache_dir": str(cache_dir),
        "n_samples_total": n_total,
        "n_audit_samples": int(n_sample),
        "n_genres": int(len(np.unique(y))),
        "finite_style_frac": float(np.mean(np.isfinite(z_style_sample))),
        "finite_content_frac": float(np.mean(np.isfinite(z_content_sample))),
        "style_latent_mean_variance": float(np.var(z_style_sample.mean(axis=0))),
        "content_latent_mean_variance": float(np.var(z_content_sample.mean(axis=0))),
        "style_probe": style_probe,
        "content_style_leakage_probe": content_style_leakage,
        "style_centroid_separation": style_separation,
        "content_track_retrieval": content_track_retrieval,
        "style_track_retrieval": style_track_retrieval,
        "direct_feature_files_present": {k: p.exists() for k, p in direct_feature_files.items()},
        "lab1_latent_files_present": {k: p.exists() for k, p in latent_files.items()},
        "production_real_music_inputs": ["mel", "donor_mel", "chroma", "onset", "beat", "genre_idx"],
        "production_bypasses_lab1_latents": production_bypasses_lab1_latents,
        "direct_features_present": direct_features_present,
        "latents_present": latents_present,
        "weak_style_latents": weak_style_latents,
        "content_identity_signal": content_identity_signal,
        "bottleneck_risk": bottleneck_risk,
        "recommendation": (
            "Keep the current real-music model on direct mel/chroma/onset/beat conditioning; do not reintroduce "
            "Lab1 z_style/z_content as the primary style/content bottleneck without replacing or retraining them."
        ),
    }
    _write_json(Path(out_path), summary)
    return summary


def completion_gate_report(
    *,
    discovery_report: Path,
    cache_dir: Path,
    train_summary: Path,
    validation_plan: Path,
    validation_pack_dir: Path,
    validation_report: Path,
    separation_report: Path,
    final_pack_dir: Path,
    listening_audit: Path,
    baseline_report: Path,
    out_path: Path,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(check_id: str, passed: bool, evidence: str, detail: Any = None) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "evidence": evidence, "detail": detail})

    discovery = _load_json(Path(discovery_report)) if Path(discovery_report).exists() else {}
    clusters = discovery.get("clusters", [])
    cluster_counts = [int(c.get("count", 0)) for c in clusters]
    add("discovery_clusters", len(clusters) >= 3 and min(cluster_counts or [0]) >= 2, str(discovery_report), {"n_clusters": len(clusters), "min_count": min(cluster_counts or [0])})

    cache_dir = Path(cache_dir)
    cache_meta = cache_dir / "diff_meta.json"
    genre_map = cache_dir / "diff_genre_to_idx.json"
    meta = _load_json(cache_meta) if cache_meta.exists() else {}
    genres = _load_json(genre_map) if genre_map.exists() else {}
    add("cache_load_artifacts", cache_meta.exists() and genre_map.exists() and int(meta.get("n_samples", 0) or 0) > 0 and len(genres) >= 3, str(cache_dir), {"n_samples": meta.get("n_samples"), "n_genres": len(genres)})

    train = _load_json(Path(train_summary)) if Path(train_summary).exists() else {}
    best = Path(str(train.get("best_checkpoint", "")))
    latest = Path(str(train.get("latest_checkpoint", "")))
    history = train.get("history", []) if isinstance(train.get("history", []), list) else []
    final_step = train.get("global_step")
    if final_step is None and history:
        final_step = history[-1].get("global_step")
    add(
        "training_checkpoints",
        best.exists() and latest.exists() and int(final_step or 0) > 1 and len(history) >= 8,
        str(train_summary),
        {"global_step": final_step, "epochs": len(history), "best_checkpoint": str(best), "latest_checkpoint": str(latest)},
    )

    plan = _load_json(Path(validation_plan)) if Path(validation_plan).exists() else {}
    pack_manifest = Path(validation_pack_dir) / "manifest.json"
    add("fixed_validation_pack", Path(validation_plan).exists() and pack_manifest.exists() and len(plan.get("rows", [])) > 0, f"{validation_plan}; {pack_manifest}", {"n_cases": len(plan.get("rows", []))})

    val = _load_json(Path(validation_report)) if Path(validation_report).exists() else {}
    metric_detail = {k: val.get(k) for k in ("mean_style_margin", "mean_content_chroma_cos", "mean_warble", "mean_fullness")}
    metric_pass = (
        Path(validation_report).exists()
        and int(val.get("n_cases", 0) or 0) > 0
        and float(val.get("mean_content_chroma_cos", 0.0) or 0.0) >= 0.75
        and float(val.get("mean_fullness", 0.0) or 0.0) >= 0.50
        and float(val.get("mean_warble", 999.0) or 999.0) <= 0.35
        and float(val.get("mean_style_margin", 0.0) or 0.0) > 0.02
    )
    add("realism_content_style_metrics", metric_pass, str(validation_report), metric_detail)

    sep = _load_json(Path(separation_report)) if Path(separation_report).exists() else {}
    sep_value = float(sep.get("mean_pairwise_target_separation", 0.0) or 0.0)
    add(
        "genre_separation",
        Path(separation_report).exists() and int(sep.get("n_targets", 0) or 0) >= 3 and sep_value > 0.001,
        str(separation_report),
        {"n_targets": sep.get("n_targets"), "mean_pairwise_target_separation": sep.get("mean_pairwise_target_separation")},
    )

    final_manifest = Path(final_pack_dir) / "manifest.csv"
    add("final_pack", final_manifest.exists(), str(final_manifest), {"exists": final_manifest.exists()})

    audit = _load_json(Path(listening_audit)) if Path(listening_audit).exists() else {}
    add("manual_listening_audit", Path(listening_audit).exists() and not bool(audit.get("manual_review_required", True)), str(listening_audit), {"manual_reviews_present": audit.get("manual_reviews_present"), "manual_review_required": audit.get("manual_review_required")})

    baseline = _load_json(Path(baseline_report)) if Path(baseline_report).exists() else {}
    add("baseline_comparison", Path(baseline_report).exists() and bool(baseline.get("baseline_available", False)), str(baseline_report), {"baseline_available": baseline.get("baseline_available")})

    summary = {"passed": bool(checks and all(c["passed"] for c in checks)), "checks": checks}
    _write_json(Path(out_path), summary)
    return summary
