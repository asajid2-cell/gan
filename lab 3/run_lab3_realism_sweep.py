#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parent

from src.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames
from src.lab3_codec_bridge import FrozenEncodec
from src.lab3_codec_data import load_codec_cache, stratified_group_split_indices as codec_group_split, stratified_split_indices as codec_split
from src.lab3_codec_judge import (
    CodecStyleJudge,
    Lab1StyleProbe,
    MERTStyleProbe,
    fit_codec_style_judge,
    fit_lab1_style_probe,
    fit_mert_style_probe,
    freeze_judge,
    freeze_mert_probe,
    freeze_probe,
)
from src.lab3_codec_models import CodecLatentTranslator, MultiScaleWaveDiscriminator
from src.lab3_codec_train import build_style_centroid_bank, build_style_exemplar_bank
from src.lab3_diffusion_data import DIFFUSION_HOP, DIFFUSION_SR, load_diffusion_cache
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import ddim_sample_v2, load_bigvgan_robust, load_checkpoint, vocode_bigvgan
from src.lab3_mert_bridge import FrozenMERT
from src.lab3_realism import (
    RealismGate,
    apply_realism_gate,
    build_balanced_transfer_plan,
    evaluate_realism_metrics,
    load_plan_audio,
    rank_realism_table,
    save_plan,
)

try:
    import soundfile as sf

    HAS_SF = True
except Exception:
    HAS_SF = False


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_lab1_checkpoint() -> Path:
    root = _repo_root()
    candidates = [
        root / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        root / "saves" / "lab1_run_combo_af_gate" / "latest.pt",
        root / "saves" / "lab1_run_a" / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_json(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _source_idx_from_index(index_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    if "source" not in index_df.columns:
        return np.zeros((len(index_df),), dtype=np.int64), {"unknown": 0}
    src_names = index_df["source"].astype(str).tolist()
    uniq = sorted(set(src_names))
    src_to_idx = {s: i for i, s in enumerate(uniq)}
    source_idx = np.asarray([src_to_idx[s] for s in src_names], dtype=np.int64)
    return source_idx, src_to_idx


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def _lab1_latents_from_audio(lab1: FrozenLab1Encoder, audio: np.ndarray, sr: int, n_frames: int) -> Dict[str, np.ndarray]:
    y = FrozenEncodec.resample_audio(np.asarray(audio, dtype=np.float32), sr_from=int(sr), sr_to=int(lab1.cfg.sample_rate))
    log_mel = extract_log_mel(y, sr=int(lab1.cfg.sample_rate))
    log_mel = fix_log_mel_frames(log_mel, n_frames=int(n_frames))
    return lab1.infer_log_mel(log_mel)


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    name = Path(path).name
    if name.startswith("epoch_"):
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 10**9, name)
    if name == "best.pt":
        return (10**9 + 1, name)
    if name == "latest.pt":
        return (10**9 + 2, name)
    if "stage1" in name:
        return (1, name)
    if "stage2" in name:
        return (2, name)
    if "stage3" in name:
        return (3, name)
    return (10**9 + 3, name)


def _resolve_codec_checkpoints(run_dir: Path, requested: Sequence[str]) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if requested:
        out = [ckpt_dir / str(x) if not Path(x).is_absolute() else Path(x) for x in requested]
    else:
        out = [p for p in [ckpt_dir / "stage1_latest.pt", ckpt_dir / "stage2_latest.pt", ckpt_dir / "stage3_latest.pt"] if p.exists()]
    return [p for p in out if p.exists()]


def _resolve_diffusion_checkpoints(run_dir: Path, requested: Sequence[str], include_all_epochs: bool) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if requested:
        out = [ckpt_dir / str(x) if not Path(x).is_absolute() else Path(x) for x in requested]
    elif include_all_epochs:
        out = sorted(ckpt_dir.glob("epoch_*.pt"), key=_checkpoint_sort_key)
        if (ckpt_dir / "best.pt").exists():
            out.append(ckpt_dir / "best.pt")
        if (ckpt_dir / "latest.pt").exists():
            out.append(ckpt_dir / "latest.pt")
    else:
        out = [p for p in [ckpt_dir / "epoch_006.pt", ckpt_dir / "best.pt", ckpt_dir / "latest.pt"] if p.exists()]
    dedup: List[Path] = []
    seen = set()
    for p in out:
        rp = str(Path(p).resolve())
        if rp not in seen and Path(p).exists():
            seen.add(rp)
            dedup.append(Path(p))
    return dedup


def _default_codec_output_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "realism_supervisor"


def _default_diffusion_output_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "realism_supervisor"


def _fit_or_load_probe(
    cache_dir: Path,
    probe_name: str,
    build_fn,
    build_kwargs: Dict,
    device: torch.device,
):
    probe_dir = Path(cache_dir) / "realism_supervisor_cache"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"{probe_name}.pt"
    meta_path = probe_dir / f"{probe_name}.json"
    if probe_path.exists() and meta_path.exists():
        meta = _load_json(meta_path)
        model_type = meta.get("type")
        if model_type == "codec_judge":
            model = CodecStyleJudge(
                in_channels=int(meta["in_channels"]),
                n_genres=int(meta["n_genres"]),
                hidden=int(meta["hidden"]),
                emb_dim=int(meta["emb_dim"]),
                n_sources=int(meta.get("n_sources", 0)),
            ).to(device)
            model.load_state_dict(torch.load(str(probe_path), map_location=device, weights_only=False), strict=True)
            return freeze_judge(model), meta
        if model_type == "lab1_probe":
            model = Lab1StyleProbe(
                in_dim=int(meta["in_dim"]),
                n_genres=int(meta["n_genres"]),
                hidden=int(meta["hidden"]),
                emb_dim=int(meta["emb_dim"]),
                n_sources=int(meta.get("n_sources", 0)),
            ).to(device)
            model.load_state_dict(torch.load(str(probe_path), map_location=device, weights_only=False), strict=True)
            return freeze_probe(model), meta
        if model_type == "mert_probe":
            model = MERTStyleProbe(
                in_dim=int(meta["in_dim"]),
                n_genres=int(meta["n_genres"]),
                hidden=int(meta["hidden"]),
                emb_dim=int(meta["emb_dim"]),
                n_sources=int(meta.get("n_sources", 0)),
            ).to(device)
            model.load_state_dict(torch.load(str(probe_path), map_location=device, weights_only=False), strict=True)
            return freeze_mert_probe(model), meta

    model, fit_result = build_fn(**build_kwargs)
    torch.save(model.state_dict(), str(probe_path))
    meta = {"fit_result": asdict(fit_result)}
    if isinstance(model, CodecStyleJudge):
        meta.update(
            {
                "type": "codec_judge",
                "in_channels": int(model.in_channels),
                "n_genres": int(model.n_genres),
                "hidden": int(model.hidden),
                "emb_dim": int(model.emb_dim),
                "n_sources": int(model.n_sources),
            }
        )
        model = freeze_judge(model)
    elif isinstance(model, Lab1StyleProbe):
        meta.update(
            {
                "type": "lab1_probe",
                "in_dim": int(model.in_dim),
                "n_genres": int(model.n_genres),
                "hidden": int(model.hidden),
                "emb_dim": int(model.emb_dim),
                "n_sources": int(model.n_sources),
            }
        )
        model = freeze_probe(model)
    else:
        meta.update(
            {
                "type": "mert_probe",
                "in_dim": int(model.in_dim),
                "n_genres": int(model.n_genres),
                "hidden": int(model.hidden),
                "emb_dim": int(model.emb_dim),
                "n_sources": int(model.n_sources),
            }
        )
        model = freeze_mert_probe(model)
    _save_json(meta, meta_path)
    return model, meta


def _build_codec_style_space(
    run_dir: Path,
    run_state: Dict,
    arrays: Dict[str, np.ndarray],
    index_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> Dict:
    cfg = run_state["config"]
    cond_source = str(cfg.get("style_cond_source", "lab1_zstyle")).strip().lower()
    n_genres = int(len(np.unique(arrays["genre_idx"])))
    source_idx, src_to_idx = _source_idx_from_index(index_df=index_df)
    n_sources = int(len(src_to_idx))

    if cond_source == "lab1_zstyle":
        emb_all = arrays["z_style"].astype(np.float32)
        cent = build_style_centroid_bank(emb_all[train_idx], arrays["genre_idx"][train_idx], n_genres=n_genres).to(device)
        ex = build_style_exemplar_bank(emb_all[train_idx], arrays["genre_idx"][train_idx], n_genres=n_genres)
        return {"name": cond_source, "centroids": cent, "exemplars": ex, "probe": None}

    if cond_source == "codec_judge_embed":
        judge, meta = _fit_or_load_probe(
            cache_dir=Path(run_dir),
            probe_name="codec_style_judge",
            build_fn=fit_codec_style_judge,
            build_kwargs={
                "arrays": arrays,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_genres": n_genres,
                "source_idx": source_idx,
                "n_sources": n_sources,
                "device": device,
                "epochs": int(cfg.get("style_judge_epochs", 20)),
                "lr": float(cfg.get("style_judge_lr", 2e-3)),
                "batch_size": int(cfg.get("style_judge_batch_size", 64)),
                "hidden": int(cfg.get("style_judge_hidden", 256)),
                "emb_dim": int(cfg.get("style_judge_emb_dim", 128)),
                "seed": int(cfg.get("seed", 328)),
                "source_adv_weight": float(cfg.get("style_judge_source_adv_weight", 0.0)),
                "source_grl_lambda": float(cfg.get("style_judge_source_grl_lambda", 1.0)),
                "grl_warmup_epochs": int(cfg.get("style_judge_grl_warmup_epochs", 0)),
                "patience": int(cfg.get("style_judge_patience", 0)),
            },
            device=device,
        )
        with torch.no_grad():
            q = torch.from_numpy(arrays["q_emb"][train_idx]).to(device).float()
            emb_all = judge.embed(q).detach().cpu().numpy().astype(np.float32)
        cent = build_style_centroid_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres).to(device)
        ex = build_style_exemplar_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres)
        return {"name": cond_source, "centroids": cent, "exemplars": ex, "probe": judge, "probe_meta": meta}

    if cond_source == "lab1_probe_embed":
        probe, meta = _fit_or_load_probe(
            cache_dir=Path(run_dir),
            probe_name="lab1_style_probe",
            build_fn=fit_lab1_style_probe,
            build_kwargs={
                "z_style": arrays["z_style"],
                "genre_idx": arrays["genre_idx"],
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_genres": n_genres,
                "source_idx": source_idx,
                "n_sources": n_sources,
                "device": device,
                "epochs": int(cfg.get("lab1_probe_epochs", 30)),
                "lr": float(cfg.get("lab1_probe_lr", 2e-3)),
                "hidden": int(cfg.get("lab1_probe_hidden", 256)),
                "emb_dim": int(cfg.get("lab1_probe_emb_dim", 128)),
                "patience": int(cfg.get("lab1_probe_patience", 8)),
                "seed": int(cfg.get("seed", 328)),
            },
            device=device,
        )
        with torch.no_grad():
            z = torch.from_numpy(arrays["z_style"][train_idx]).to(device).float()
            emb_all = probe.embed(z).detach().cpu().numpy().astype(np.float32)
        cent = build_style_centroid_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres).to(device)
        ex = build_style_exemplar_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres)
        return {"name": cond_source, "centroids": cent, "exemplars": ex, "probe": probe, "probe_meta": meta}

    if cond_source == "mert_probe_embed":
        if "mert_feat" not in arrays:
            raise RuntimeError("Codec cache does not contain mert_feat, but run requires mert_probe_embed.")
        probe, meta = _fit_or_load_probe(
            cache_dir=Path(run_dir),
            probe_name="mert_style_probe",
            build_fn=fit_mert_style_probe,
            build_kwargs={
                "mert_feat": arrays["mert_feat"],
                "genre_idx": arrays["genre_idx"],
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_genres": n_genres,
                "source_idx": source_idx,
                "n_sources": n_sources,
                "device": device,
                "epochs": int(cfg.get("mert_probe_epochs", 30)),
                "lr": float(cfg.get("mert_probe_lr", 2e-3)),
                "hidden": int(cfg.get("mert_probe_hidden", 256)),
                "emb_dim": int(cfg.get("mert_probe_emb_dim", 128)),
                "patience": int(cfg.get("mert_probe_patience", 8)),
                "seed": int(cfg.get("seed", 328)),
                "source_adv_weight": float(cfg.get("style_judge_source_adv_weight", 0.0)),
                "source_grl_lambda": float(cfg.get("style_judge_source_grl_lambda", 1.0)),
                "grl_warmup_epochs": int(cfg.get("style_judge_grl_warmup_epochs", 0)),
            },
            device=device,
        )
        with torch.no_grad():
            feat = torch.from_numpy(arrays["mert_feat"][train_idx]).to(device).float()
            emb_all = probe.embed(feat).detach().cpu().numpy().astype(np.float32)
        cent = build_style_centroid_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres).to(device)
        ex = build_style_exemplar_bank(emb_all, arrays["genre_idx"][train_idx], n_genres=n_genres)
        return {"name": cond_source, "centroids": cent, "exemplars": ex, "probe": probe, "probe_meta": meta}

    raise NotImplementedError(f"Unsupported codec style_cond_source for realism sweep: {cond_source}")


def _codec_condition_for_target(
    style_space: Dict,
    target_idx: int,
    cond_mode: str,
    cond_alpha: float,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    cent = style_space["centroids"][target_idx : target_idx + 1].to(device).float()
    ex_bank = style_space["exemplars"].get(int(target_idx)) if style_space.get("exemplars") is not None else None
    if ex_bank is None or len(ex_bank) == 0:
        ex = cent
    else:
        i = int(rng.integers(0, int(ex_bank.shape[0])))
        ex = ex_bank[i : i + 1].to(device).float()
    mode = str(cond_mode).strip().lower()
    if mode == "centroid":
        z = cent
    elif mode == "exemplar":
        z = ex
    else:
        z = float(cond_alpha) * cent + (1.0 - float(cond_alpha)) * ex
    return F.normalize(z, dim=-1)


def _codec_style_metrics(
    style_space: Dict,
    device: torch.device,
    generated_z_style: List[np.ndarray],
    generated_audio: Sequence[np.ndarray],
    generated_qhat_embed: List[np.ndarray],
    target_idx: np.ndarray,
    mert: Optional[FrozenMERT],
    sr: int,
) -> Dict[str, float]:
    centroids = style_space["centroids"].detach().cpu().numpy().astype(np.float32)
    source = style_space["name"]
    if source == "codec_judge_embed":
        emb = np.stack(generated_qhat_embed).astype(np.float32)
    elif source == "lab1_zstyle":
        emb = np.stack(generated_z_style).astype(np.float32)
    elif source == "lab1_probe_embed":
        probe = style_space["probe"]
        z = torch.from_numpy(np.stack(generated_z_style).astype(np.float32)).to(device)
        with torch.no_grad():
            emb = probe.embed(z).detach().cpu().numpy().astype(np.float32)
    elif source == "mert_probe_embed":
        if mert is None:
            return {"style_target_acc": float("nan"), "style_target_cos": float("nan")}
        probe = style_space["probe"]
        fake_feat = []
        for wav in generated_audio:
            fake_feat.append(FrozenMERT.resample_audio(np.asarray(wav, dtype=np.float32), sr_from=int(sr), sr_to=int(mert.cfg.sample_rate)))
        with torch.no_grad():
            raw = mert.extract_features_batch(fake_feat)
            emb = probe.embed(torch.from_numpy(raw).to(device).float()).detach().cpu().numpy().astype(np.float32)
    else:
        return {"style_target_acc": float("nan"), "style_target_cos": float("nan")}

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    cent = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = emb @ cent.T
    tgt = np.asarray(target_idx, dtype=np.int64)
    pred = np.argmax(sim, axis=1)
    return {
        "style_target_acc": float(np.mean(pred == tgt)),
        "style_target_cos": float(np.mean(sim[np.arange(len(tgt)), tgt])),
    }


def _run_codec_sweep(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir or _default_codec_output_dir(run_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_state = _load_json(run_dir / "run_state.json")
    cfg = run_state["config"]
    device = _device_from_arg(args.device)
    print(f"[codec-sweep] device={device}")

    cache_dir = Path(cfg.get("reuse_cache_dir") or (run_dir / "cache"))
    index_df, arrays, _genre_to_idx, meta = load_codec_cache(cache_dir)
    genre_idx = np.asarray(arrays["genre_idx"])
    if bool(cfg.get("split_by_track", True)) and "track_id" in index_df.columns:
        train_idx, val_idx = codec_group_split(
            genre_idx,
            index_df["track_id"].to_numpy(),
            val_ratio=float(cfg.get("val_ratio", 0.15)),
            seed=int(cfg.get("seed", 328)),
        )
    else:
        train_idx, val_idx = codec_split(
            genre_idx,
            val_ratio=float(cfg.get("val_ratio", 0.15)),
            seed=int(cfg.get("seed", 328)),
        )

    plan = build_balanced_transfer_plan(
        index_df=index_df,
        genre_idx=genre_idx,
        val_idx=val_idx,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
    )
    save_plan(plan, out_dir / "transfer_plan.csv")
    plan_audio = load_plan_audio(plan=plan, sample_rate=int(meta.codec_sample_rate), chunk_seconds=float(meta.codec_chunk_seconds))

    lab1 = FrozenLab1Encoder(Path(cfg.get("lab1_checkpoint", _default_lab1_checkpoint())), device=str(device))
    codec = FrozenEncodec(
        model_id=str(cfg.get("codec_model_id", "facebook/encodec_24khz")),
        bandwidth=float(cfg.get("codec_bandwidth", 6.0)),
        chunk_seconds=float(cfg.get("codec_chunk_seconds", 5.0)),
        device=str(device),
    )
    mert = FrozenMERT(
        model_id=str(cfg.get("mert_model_id", "m-a-p/MERT-v1-95M")),
        chunk_seconds=float(meta.codec_chunk_seconds),
        device=str(device),
        layer=int(cfg.get("mert_layer", -1)),
    )
    style_space = _build_codec_style_space(run_dir, run_state, arrays, index_df, train_idx, val_idx, device)

    gen = CodecLatentTranslator(
        in_channels=int(meta.codec_channels),
        z_content_dim=int(arrays["z_content"].shape[1]),
        z_style_dim=int(style_space["centroids"].shape[1]),
        hidden_channels=int(cfg.get("translator_hidden_channels", 256)),
        n_blocks=int(cfg.get("translator_blocks", 10)),
        noise_dim=int(cfg.get("translator_noise_dim", 32)),
        residual_scale=float(cfg.get("translator_residual_scale", 0.5)),
        direct_output=bool(cfg.get("translator_direct_output", False)),
        direct_mix=float(cfg.get("translator_direct_mix", 1.0)),
    ).to(device)
    disc = MultiScaleWaveDiscriminator(n_scales=int(cfg.get("discriminator_scales", 3))).to(device)
    checkpoints = _resolve_codec_checkpoints(run_dir=run_dir, requested=args.checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"No codec checkpoints found under {run_dir / 'checkpoints'}")

    gate = RealismGate(
        max_fad_mert=args.max_fad_mert,
        max_target_centroid_mae_norm=args.max_target_centroid_mae_norm,
        max_target_hf_mae=args.max_target_hf_mae,
        max_target_lf_mae=args.max_target_lf_mae,
        max_target_dynamic_range_mae_db=args.max_target_dynamic_range_mae_db,
        min_mps=args.min_mps,
        min_style_target_acc=args.min_style_target_acc,
        min_style_target_cos=args.min_style_target_cos,
    )

    rows: List[Dict] = []
    rng = np.random.default_rng(int(args.seed))
    cond_mode_map = {
        "stage1": ("centroid", 1.0),
        "stage2": (str(cfg.get("stage2_cond_mode", "exemplar")), float(cfg.get("stage2_cond_alpha_end", 0.4))),
        "stage3": (str(cfg.get("stage3_cond_mode", "exemplar")), float(cfg.get("stage3_cond_alpha_end", 0.2))),
    }

    for ckpt_path in checkpoints:
        print(f"[codec-sweep] evaluating {ckpt_path.name}")
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        gen.load_state_dict(payload["generator"], strict=True)
        disc.load_state_dict(payload["discriminator"], strict=True)
        gen.eval()
        stage_tag = str(payload.get("meta", {}).get("stage", "stage3")).lower()
        cond_mode, cond_alpha = cond_mode_map.get(stage_tag, cond_mode_map["stage3"])

        fake_audio: List[np.ndarray] = []
        gen_z_style: List[np.ndarray] = []
        qhat_emb: List[np.ndarray] = []
        mps_vals: List[float] = []

        ex_dir = out_dir / "examples" / ckpt_path.stem
        if int(args.write_audio_count) > 0:
            ex_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(plan.itertuples(index=False)):
            src_row = int(row.source_row)
            tgt_idx = int(row.target_genre_idx)
            q_src = torch.from_numpy(arrays["q_emb"][src_row : src_row + 1]).to(device).float()
            zc = torch.from_numpy(arrays["z_content"][src_row : src_row + 1]).to(device).float()
            z_tgt = _codec_condition_for_target(style_space, tgt_idx, cond_mode, float(cond_alpha), device, rng)

            with torch.no_grad():
                q_hat = gen(q_src=q_src, z_content=zc, z_style_tgt=z_tgt)
                wav = codec.decode_embeddings(q_hat)[0, 0].detach().cpu().numpy().astype(np.float32)
            wav = wav / (np.max(np.abs(wav)) + 1e-8)
            fake_audio.append(wav)
            lat = _lab1_latents_from_audio(lab1=lab1, audio=wav, sr=int(meta.codec_sample_rate), n_frames=int(meta.lab1_n_frames))
            gen_z_style.append(lat["z_style"])
            mps_vals.append(_cosine_np(lat["z_content"], arrays["z_content"][src_row]))
            if style_space["name"] == "codec_judge_embed":
                with torch.no_grad():
                    qhat_emb.append(style_space["probe"].embed(q_hat).detach().cpu().numpy()[0].astype(np.float32))

            if HAS_SF and idx < int(args.write_audio_count):
                sf.write(str(ex_dir / f"{idx:02d}_gen.wav"), wav, int(meta.codec_sample_rate))
                sf.write(str(ex_dir / f"{idx:02d}_source.wav"), plan_audio["source_audio"][idx], int(meta.codec_sample_rate))
                sf.write(str(ex_dir / f"{idx:02d}_target_ref.wav"), plan_audio["target_ref_audio"][idx], int(meta.codec_sample_rate))

        realism = evaluate_realism_metrics(
            plan=plan,
            fake_audio=fake_audio,
            target_ref_audio=plan_audio["target_ref_audio"],
            sr=int(meta.codec_sample_rate),
            mert=mert,
            mert_batch_size=int(args.mert_batch_size),
        )
        style_metrics = _codec_style_metrics(
            style_space=style_space,
            device=device,
            generated_z_style=gen_z_style,
            generated_audio=fake_audio,
            generated_qhat_embed=qhat_emb,
            target_idx=plan["target_genre_idx"].to_numpy(),
            mert=mert,
            sr=int(meta.codec_sample_rate),
        )
        row_out = {
            "checkpoint": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "stage": stage_tag,
            "n_samples": int(len(plan)),
            "mps": float(np.mean(mps_vals)),
            **style_metrics,
            **realism,
        }
        checks = apply_realism_gate(row=row_out, gate=gate)
        row_out.update({f"pass_{k}": bool(v) for k, v in checks.items()})
        rows.append(row_out)

    df = rank_realism_table(pd.DataFrame(rows))
    df.to_csv(out_dir / "codec_realism_sweep.csv", index=False)
    best = df.iloc[0].to_dict()
    _save_json({"best": best, "gate": asdict(gate), "plan_csv": str(out_dir / "transfer_plan.csv")}, out_dir / "codec_realism_best.json")
    print(df[["rank", "checkpoint", "fad_mert", "target_hf_mae", "target_dynamic_range_mae_db", "mps", "style_target_acc"]].to_string(index=False))
    print(f"[codec-sweep] summary={out_dir / 'codec_realism_sweep.csv'}")


def _diffusion_build_cond(arrays: Dict[str, np.ndarray], row_idx: int, max_frames: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = int(max_frames)
    chroma = np.array(arrays["chroma"][row_idx], copy=True).astype(np.float32)[:, :t]
    onset = np.array(arrays["onset"][row_idx], copy=True).astype(np.float32)[:t]
    beat = np.array(arrays["beat"][row_idx], copy=True).astype(np.float32)[:t]
    chroma_t = torch.from_numpy(chroma).unsqueeze(1).expand(-1, 80, -1)
    onset_t = torch.from_numpy(onset).reshape(1, 1, t).expand(1, 80, -1)
    beat_t = torch.from_numpy(beat).reshape(1, 1, t).expand(1, 80, -1)
    cond_feat = torch.cat([chroma_t, onset_t, beat_t], dim=0).contiguous()
    z_content = torch.from_numpy(np.array(arrays["z_content"][row_idx], copy=True).astype(np.float32))
    z_style = torch.from_numpy(np.array(arrays["z_style"][row_idx], copy=True).astype(np.float32))
    return cond_feat, z_content, z_style


def _run_diffusion_sweep(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir or _default_diffusion_output_dir(run_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_json(run_dir / "v2_config.json")
    device = _device_from_arg(args.device)
    print(f"[diffusion-sweep] device={device}")

    cache_dir = Path(cfg["cache_dir"])
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"])
    track_ids = index_df["track_id"].to_numpy()
    from src.lab3_data import stratified_group_split_indices
    train_idx, val_idx = stratified_group_split_indices(genre_idx, track_ids, val_ratio=0.1, seed=int(cfg.get("seed", 328)))

    max_frames = int(cfg.get("max_frames", 256))
    eff_sec = float(max_frames * DIFFUSION_HOP / DIFFUSION_SR)
    plan = build_balanced_transfer_plan(
        index_df=index_df,
        genre_idx=genre_idx,
        val_idx=val_idx,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
    )
    save_plan(plan, out_dir / "transfer_plan.csv")
    plan_audio = load_plan_audio(plan=plan, sample_rate=int(DIFFUSION_SR), chunk_seconds=float(eff_sec))

    lab1 = FrozenLab1Encoder(_default_lab1_checkpoint(), device=str(device))
    mert = FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=float(eff_sec), device=str(device), layer=-1)
    style_centroids = build_style_centroid_bank(
        np.asarray(arrays["z_style"][train_idx]).astype(np.float32),
        np.asarray(arrays["genre_idx"][train_idx]).astype(np.int64),
        n_genres=int(len(genre_to_idx)),
    ).numpy().astype(np.float32)

    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=int(cfg.get("base_ch", 64)),
        ch_mults=tuple(cfg.get("ch_mults", [1, 2, 4, 4])),
        n_res=int(cfg.get("n_res", 2)),
        attn_levels=tuple(cfg.get("attn_levels", [2, 3])),
        z_content_dim=128,
        z_style_dim=128,
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9999)))

    vocoder = load_bigvgan_robust(device=device)

    checkpoints = _resolve_diffusion_checkpoints(run_dir=run_dir, requested=args.checkpoints, include_all_epochs=bool(args.include_all_epochs))
    if not checkpoints:
        raise FileNotFoundError(f"No diffusion checkpoints found under {run_dir / 'checkpoints'}")

    gate = RealismGate(
        max_fad_mert=args.max_fad_mert,
        max_target_centroid_mae_norm=args.max_target_centroid_mae_norm,
        max_target_hf_mae=args.max_target_hf_mae,
        max_target_lf_mae=args.max_target_lf_mae,
        max_target_dynamic_range_mae_db=args.max_target_dynamic_range_mae_db,
        min_mps=args.min_mps,
        min_style_target_acc=args.min_style_target_acc,
        min_style_target_cos=args.min_style_target_cos,
    )

    rows: List[Dict] = []
    for ckpt_path in checkpoints:
        print(f"[diffusion-sweep] evaluating {ckpt_path.name}")
        try:
            ckpt = load_checkpoint(ckpt_path, model, ema, device=device)
        except Exception as e:
            rows.append(
                {
                    "checkpoint": ckpt_path.name,
                    "checkpoint_path": str(ckpt_path),
                    "epoch": -1,
                    "n_samples": int(len(plan)),
                    "error": str(e),
                    "pass_all": False,
                }
            )
            print(f"[diffusion-sweep] skip {ckpt_path.name}: {e}")
            continue
        fake_audio: List[np.ndarray] = []
        gen_z_style: List[np.ndarray] = []
        mps_vals: List[float] = []

        ex_dir = out_dir / "examples" / ckpt_path.stem
        if int(args.write_audio_count) > 0:
            ex_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(plan.itertuples(index=False)):
            src_row = int(row.source_row)
            tgt_row = int(row.target_ref_row)
            cond_feat, z_content, _ = _diffusion_build_cond(arrays=arrays, row_idx=src_row, max_frames=max_frames)
            _, _, z_style = _diffusion_build_cond(arrays=arrays, row_idx=tgt_row, max_frames=max_frames)
            cond_feat = cond_feat.unsqueeze(0).to(device).float()
            z_content = z_content.unsqueeze(0).to(device).float()
            z_style = z_style.unsqueeze(0).to(device).float()

            with torch.no_grad():
                mel_gen = ddim_sample_v2(
                    ema.shadow,
                    schedule,
                    cond_feat,
                    z_content,
                    z_style,
                    n_steps=int(args.ddim_steps),
                    guidance_scale=float(args.guidance_scale if args.guidance_scale is not None else cfg.get("guidance_scale", 2.0)),
                    device=device,
                )
                wav = vocode_bigvgan(mel_gen, meta.mel_min, meta.mel_max, vocoder, device)[0].astype(np.float32)
            wav = wav / (np.max(np.abs(wav)) + 1e-8)
            fake_audio.append(wav)
            lat = _lab1_latents_from_audio(lab1=lab1, audio=wav, sr=int(DIFFUSION_SR), n_frames=256)
            gen_z_style.append(lat["z_style"])
            mps_vals.append(_cosine_np(lat["z_content"], np.asarray(arrays["z_content"][src_row], dtype=np.float32)))

            if HAS_SF and idx < int(args.write_audio_count):
                sf.write(str(ex_dir / f"{idx:02d}_gen.wav"), wav, int(DIFFUSION_SR))
                sf.write(str(ex_dir / f"{idx:02d}_source.wav"), plan_audio["source_audio"][idx], int(DIFFUSION_SR))
                sf.write(str(ex_dir / f"{idx:02d}_target_ref.wav"), plan_audio["target_ref_audio"][idx], int(DIFFUSION_SR))

        realism = evaluate_realism_metrics(
            plan=plan,
            fake_audio=fake_audio,
            target_ref_audio=plan_audio["target_ref_audio"],
            sr=int(DIFFUSION_SR),
            mert=mert,
            mert_batch_size=int(args.mert_batch_size),
        )
        gen_style = np.stack(gen_z_style).astype(np.float32)
        gen_style = gen_style / (np.linalg.norm(gen_style, axis=1, keepdims=True) + 1e-8)
        cent = style_centroids / (np.linalg.norm(style_centroids, axis=1, keepdims=True) + 1e-8)
        sim = gen_style @ cent.T
        tgt = plan["target_genre_idx"].to_numpy(dtype=np.int64)
        style_metrics = {
            "style_target_acc": float(np.mean(np.argmax(sim, axis=1) == tgt)),
            "style_target_cos": float(np.mean(sim[np.arange(len(tgt)), tgt])),
        }
        row_out = {
            "checkpoint": ckpt_path.name,
            "checkpoint_path": str(ckpt_path),
            "epoch": int(ckpt.get("epoch", -1)),
            "n_samples": int(len(plan)),
            "mps": float(np.mean(mps_vals)),
            **style_metrics,
            **realism,
        }
        checks = apply_realism_gate(row=row_out, gate=gate)
        row_out.update({f"pass_{k}": bool(v) for k, v in checks.items()})
        rows.append(row_out)

    df = rank_realism_table(pd.DataFrame(rows))
    df.to_csv(out_dir / "diffusion_realism_sweep.csv", index=False)
    best = df.iloc[0].to_dict()
    _save_json({"best": best, "gate": asdict(gate), "plan_csv": str(out_dir / "transfer_plan.csv")}, out_dir / "diffusion_realism_best.json")
    print(df[["rank", "checkpoint", "fad_mert", "target_hf_mae", "target_dynamic_range_mae_db", "mps", "style_target_acc"]].to_string(index=False))
    print(f"[diffusion-sweep] summary={out_dir / 'diffusion_realism_sweep.csv'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realism-first checkpoint supervisor for DGGR codec and diffusion runs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(pp):
        pp.add_argument("--run-dir", type=Path, required=True)
        pp.add_argument("--output-dir", type=Path, default=None)
        pp.add_argument("--checkpoints", nargs="*", default=[])
        pp.add_argument("--n-samples", type=int, default=24)
        pp.add_argument("--seed", type=int, default=328)
        pp.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        pp.add_argument("--mert-batch-size", type=int, default=8)
        pp.add_argument("--write-audio-count", type=int, default=4)
        pp.add_argument("--max-fad-mert", type=float, default=None)
        pp.add_argument("--max-target-centroid-mae-norm", type=float, default=None)
        pp.add_argument("--max-target-hf-mae", type=float, default=None)
        pp.add_argument("--max-target-lf-mae", type=float, default=None)
        pp.add_argument("--max-target-dynamic-range-mae-db", type=float, default=None)
        pp.add_argument("--min-mps", type=float, default=None)
        pp.add_argument("--min-style-target-acc", type=float, default=None)
        pp.add_argument("--min-style-target-cos", type=float, default=None)

    p_codec = sub.add_parser("codec", help="Sweep codec checkpoints for realism.")
    _add_common(p_codec)

    p_diff = sub.add_parser("diffusion", help="Sweep diffusion checkpoints for realism.")
    _add_common(p_diff)
    p_diff.add_argument("--include-all-epochs", action="store_true")
    p_diff.add_argument("--ddim-steps", type=int, default=50)
    p_diff.add_argument("--guidance-scale", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "codec":
        _run_codec_sweep(args)
    else:
        _run_diffusion_sweep(args)


if __name__ == "__main__":
    main()
