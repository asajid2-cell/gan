from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor

from src.lab3_data import DEFAULT_MANIFESTS, genre_source_table, load_manifests


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-label manifests with genre predictions (CLAP zero-shot) to decouple genre from dataset source."
    )
    p.add_argument("--manifests-root", type=Path, default=Path("Z:/DataSets/_lab1_manifests"))
    p.add_argument("--manifest-files", nargs="*", default=DEFAULT_MANIFESTS)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-audit-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)

    p.add_argument("--labels", nargs="+", required=True, help="Genre label set (e.g., hiphop lofi classical electronic).")
    p.add_argument("--prompt-template", type=str, default="a {label} music track")
    p.add_argument("--min-conf", type=float, default=0.0, help="If >0, low-confidence items become 'unassigned'.")

    p.add_argument("--chunk-seconds", type=float, default=8.0)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--sr", type=int, default=48000, help="CLAP models typically use 48kHz audio.")
    p.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--model-id", type=str, default="laion/clap-htsat-fused")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-files", type=int, default=0, help="0 means all")
    p.add_argument("--seed", type=int, default=328)
    return p.parse_args()


def _pick_device(arg: str) -> str:
    a = str(arg).strip().lower()
    if a == "cpu":
        return "cpu"
    if a == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_load_audio(path: str, sr: int, mono: bool, start_sec: float, chunk_seconds: float) -> np.ndarray:
    try:
        y, _ = librosa.load(
            path,
            sr=int(sr),
            mono=bool(mono),
            offset=float(max(0.0, start_sec)),
            duration=float(max(0.1, chunk_seconds)),
        )
        if y is None:
            return np.zeros((0,), dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1:
            y = np.mean(y, axis=0).astype(np.float32)
        return y
    except Exception:
        return np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def _encode_text(
    processor: ClapProcessor, model: ClapModel, prompts: List[str], device: str
) -> torch.Tensor:
    inp = processor(text=prompts, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    feat = model.get_text_features(**inp)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat


@torch.no_grad()
def _predict_batch(
    processor: ClapProcessor,
    model: ClapModel,
    text_feat: torch.Tensor,
    audio_list: List[np.ndarray],
    device: str,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # CLAP processor accepts raw audio arrays.
    inputs = processor(audios=audio_list, sampling_rate=int(sr), return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    a_feat = model.get_audio_features(**inputs)
    a_feat = torch.nn.functional.normalize(a_feat, dim=-1)
    logits = a_feat @ text_feat.T
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return idx.detach().cpu().numpy().astype(np.int64), conf.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))
    device = _pick_device(args.device)

    df = load_manifests(Path(args.manifests_root), manifest_files=args.manifest_files)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    if int(args.max_files) > 0:
        n = min(int(args.max_files), len(df))
        df = df.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    labels = [str(x).strip() for x in args.labels if str(x).strip()]
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels.")
    prompts = [str(args.prompt_template).format(label=l) for l in labels]

    print(f"[auto-genre] device={device} model={args.model_id} n_files={len(df)} n_labels={len(labels)}")
    processor = ClapProcessor.from_pretrained(str(args.model_id))
    model = ClapModel.from_pretrained(str(args.model_id)).to(device)
    model.eval()

    text_feat = _encode_text(processor, model, prompts=prompts, device=device)

    pred_idx: List[int] = []
    pred_conf: List[float] = []
    pred_err: List[int] = []

    bs = max(1, int(args.batch_size))
    for i in range(0, len(df), bs):
        batch = df.iloc[i : i + bs]
        audio_list: List[np.ndarray] = []
        ok_mask: List[bool] = []
        for p in batch["path"].astype(str).tolist():
            y = _safe_load_audio(
                p,
                sr=int(args.sr),
                mono=bool(args.mono),
                start_sec=float(args.start_sec),
                chunk_seconds=float(args.chunk_seconds),
            )
            ok = bool(y.shape[0] > 0)
            ok_mask.append(ok)
            audio_list.append(y if ok else np.zeros((int(args.sr * 0.25),), dtype=np.float32))

        idx, conf = _predict_batch(
            processor=processor,
            model=model,
            text_feat=text_feat,
            audio_list=audio_list,
            device=device,
            sr=int(args.sr),
        )
        for j in range(len(batch)):
            if not ok_mask[j]:
                pred_idx.append(-1)
                pred_conf.append(0.0)
                pred_err.append(1)
            else:
                pred_idx.append(int(idx[j]))
                pred_conf.append(float(conf[j]))
                pred_err.append(0)

        if (i // bs) % 10 == 0:
            done = min(i + bs, len(df))
            print(f"[auto-genre] {done}/{len(df)}")

    out = df.copy()
    out["genre_pred_idx"] = np.array(pred_idx, dtype=np.int64)
    out["genre_conf"] = np.array(pred_conf, dtype=np.float32)
    out["genre_error"] = np.array(pred_err, dtype=np.int64)
    out["genre_model"] = str(args.model_id)
    out["genre_prompt_template"] = str(args.prompt_template)

    genres: List[str] = []
    min_conf = float(args.min_conf)
    for k, c, e in zip(out["genre_pred_idx"].tolist(), out["genre_conf"].tolist(), out["genre_error"].tolist()):
        if int(e) != 0:
            genres.append("unassigned")
            continue
        if int(k) < 0 or int(k) >= len(labels):
            genres.append("unassigned")
            continue
        if min_conf > 0.0 and float(c) < min_conf:
            genres.append("unassigned")
            continue
        genres.append(str(labels[int(k)]))
    out["genre"] = genres

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[auto-genre] wrote {out_csv}")

    audit = genre_source_table(out[out["genre"] != "unassigned"].copy())
    if args.out_audit_csv is None:
        out_audit = out_csv.with_suffix("").with_name(out_csv.stem + "_genre_source_table.csv")
    else:
        out_audit = Path(args.out_audit_csv)
    audit.to_csv(out_audit, index=False)
    print(f"[auto-genre] wrote {out_audit}")

    if args.out_json is None:
        out_json = out_csv.with_suffix("").with_name(out_csv.stem + "_info.json")
    else:
        out_json = Path(args.out_json)
    info = {
        "model_id": str(args.model_id),
        "labels": labels,
        "prompt_template": str(args.prompt_template),
        "min_conf": float(args.min_conf),
        "n_rows": int(len(out)),
        "n_assigned": int(int((out["genre"] != "unassigned").sum())),
        "error_rate": float(np.mean(out["genre_error"].to_numpy().astype(np.float64))),
        "mean_conf_assigned": float(out.loc[out["genre"] != "unassigned", "genre_conf"].mean() if int((out["genre"] != "unassigned").sum()) > 0 else float("nan")),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[auto-genre] wrote {out_json}")


if __name__ == "__main__":
    main()

