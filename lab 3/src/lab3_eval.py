from __future__ import annotations

from typing import Dict, Optional

import librosa
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from .lab3_bridge import FrozenLab1Encoder, denormalize_log_mel
from .lab3_losses import cosine_distance, multi_resolution_stft_loss
from .lab3_models import ReconstructionDecoder


def _sample_shift_targets(source_idx: torch.Tensor, n_genres: int) -> torch.Tensor:
    tgt = torch.randint(low=0, high=int(n_genres), size=source_idx.shape, device=source_idx.device)
    clash = tgt == source_idx
    if clash.any():
        tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt


def fit_third_party_style_classifier(z_style_train: np.ndarray, genre_idx_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1500, random_state=328, n_jobs=1)
    clf.fit(z_style_train, genre_idx_train)
    return clf


def _pairwise_cos_mean(feats: np.ndarray) -> float:
    if feats.shape[0] < 2:
        return float("nan")
    x = feats.astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


@torch.no_grad()
def evaluate_lab1_style_judge_quality(
    frozen_encoder: FrozenLab1Encoder,
    val_loader: DataLoader,
    genre_to_source_idx: torch.Tensor,
    device: str,
    max_batches: Optional[int] = None,
) -> Dict:
    genre_to_source_idx = genre_to_source_idx.to(device).long()
    hit_vals = []
    conf_vals = []

    for bidx, batch in enumerate(val_loader):
        if max_batches is not None and bidx >= int(max_batches):
            break
        mel_real = batch["mel_norm"].to(device).float()
        gidx = batch["genre_idx"].to(device).long()
        real_db = denormalize_log_mel(mel_real)
        enc = frozen_encoder.forward_log_mel_tensor(real_db)
        logits = enc["style_logits"]
        tgt_source = genre_to_source_idx[gidx]
        valid = tgt_source >= 0
        if not bool(valid.any()):
            continue
        probs = torch.softmax(logits[valid], dim=1)
        tgt_valid = tgt_source[valid]
        pred = torch.argmax(probs, dim=1)
        conf = probs[torch.arange(len(tgt_valid), device=probs.device), tgt_valid]
        hit_vals.append((pred == tgt_valid).float().cpu().numpy())
        conf_vals.append(conf.float().cpu().numpy())

    if not hit_vals:
        return {
            "style_judge_mode": "lab1_head",
            "n_eval": 0,
            "style_judge_val_acc": float("nan"),
            "style_judge_val_conf": float("nan"),
        }

    return {
        "style_judge_mode": "lab1_head",
        "n_eval": int(np.sum([len(x) for x in hit_vals])),
        "style_judge_val_acc": float(np.concatenate(hit_vals).mean()),
        "style_judge_val_conf": float(np.concatenate(conf_vals).mean()),
    }


@torch.no_grad()
def evaluate_genre_shift(
    generator: ReconstructionDecoder,
    frozen_encoder: FrozenLab1Encoder,
    val_loader: DataLoader,
    cond_bank: torch.Tensor,
    style_classifier: Optional[LogisticRegression],
    genre_to_source_idx: Optional[torch.Tensor],
    device: str,
    style_judge_mode: str = "lab1_head",
    mps_threshold: float = 0.90,
    style_acc_threshold: float = 0.75,
    sf_threshold: float = 0.85,
    continuity_max: float = 4.5,
    hf_lf_ratio_max: float = 1.2,
    spectral_centroid_min_hz: float = 300.0,
    spectral_centroid_max_hz: float = 4000.0,
    max_fake_pairwise_cos: float = 0.92,
    max_batches: Optional[int] = None,
    idx_to_genre: Optional[Dict[int, str]] = None,
) -> Dict:
    generator.eval()
    n_genres = int(cond_bank.shape[0])
    cond_bank = cond_bank.to(device)
    if genre_to_source_idx is not None:
        genre_to_source_idx = genre_to_source_idx.to(device).long()

    mps_vals = []
    sf_conf_vals = []
    sf_margin_vals = []
    sf_entropy_adj_vals = []
    sf_pred_hits = []
    continuity_vals = []
    fake_centroid_vals = []
    fake_hf_lf_ratio_vals = []
    fake_collapse_feats = []
    per_target: Dict[int, Dict[str, list]] = {}
    n = 0

    first_batch = next(iter(val_loader))
    n_mels = int(first_batch["mel_norm"].shape[1])
    mel_freqs = librosa.mel_frequencies(
        n_mels=n_mels,
        fmin=20,
        fmax=float(frozen_encoder.cfg.sample_rate / 2.0),
    ).astype(np.float32)
    hf_bins = max(1, int(round(len(mel_freqs) * 0.20)))
    lf_bins = max(1, int(round(len(mel_freqs) * 0.20)))

    for bidx, batch in enumerate(val_loader):
        if max_batches is not None and bidx >= int(max_batches):
            break
        mel_real = batch["mel_norm"].to(device).float()
        zc = batch["z_content"].to(device).float()
        gidx = batch["genre_idx"].to(device).long()
        tgt_idx = _sample_shift_targets(gidx, n_genres=n_genres)
        cond = cond_bank[tgt_idx]

        fake = generator(zc, cond)
        fake_db = denormalize_log_mel(fake)
        enc_out = frozen_encoder.forward_log_mel_tensor(fake_db)
        zc_p = enc_out["z_content"]
        zs_p = enc_out["z_style"]

        mps_batch = 1.0 - cosine_distance(zc_p, zc)
        mps_vals.append(mps_batch.detach().cpu().numpy())

        cont = multi_resolution_stft_loss(fake, mel_real)
        continuity_vals.append(float(cont.item()))

        if str(style_judge_mode) == "lab1_head":
            if genre_to_source_idx is None:
                raise ValueError("genre_to_source_idx is required for style_judge_mode='lab1_head'.")
            style_logits = enc_out["style_logits"]
            tgt_source = genre_to_source_idx[tgt_idx]
            valid = tgt_source >= 0
            if bool(valid.any()):
                probs_t = torch.softmax(style_logits[valid], dim=1)
                tgt_valid = tgt_source[valid]
                conf_t = probs_t[torch.arange(len(tgt_valid), device=probs_t.device), tgt_valid]
                pred_t = torch.argmax(probs_t, dim=1)
                probs = probs_t.detach().cpu().numpy()
                conf = conf_t.detach().cpu().numpy()
                pred = pred_t.detach().cpu().numpy()
                tgt_np = tgt_valid.detach().cpu().numpy()
                if probs.shape[1] > 1:
                    second = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2]
                    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / np.log(float(probs.shape[1]))
                else:
                    second = np.zeros_like(conf)
                    entropy = np.zeros_like(conf)
                sf_conf_vals.append(conf.astype(np.float32))
                sf_margin_vals.append((conf - second).astype(np.float32))
                sf_entropy_adj_vals.append((conf * (1.0 - entropy)).astype(np.float32))
                sf_pred_hits.append((pred == tgt_np).astype(np.float32))
                for i_loc in range(len(tgt_np)):
                    tg = int(tgt_np[i_loc])
                    rec = per_target.setdefault(
                        tg,
                        {"conf": [], "margin": [], "entropy_adj": [], "hit": [], "centroid": [], "hf_lf": []},
                    )
                    rec["conf"].append(float(conf[i_loc]))
                    rec["margin"].append(float(conf[i_loc] - second[i_loc]))
                    rec["entropy_adj"].append(float(conf[i_loc] * (1.0 - entropy[i_loc])))
                    rec["hit"].append(float(pred[i_loc] == tgt_np[i_loc]))
        else:
            if style_classifier is None:
                raise ValueError("style_classifier is required for style_judge_mode='logreg_train'.")
            zs_np = zs_p.detach().cpu().numpy()
            probs = style_classifier.predict_proba(zs_np)
            tgt_np = tgt_idx.detach().cpu().numpy()
            conf = probs[np.arange(len(tgt_np)), tgt_np]
            if probs.shape[1] > 1:
                second = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2]
                entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / np.log(float(probs.shape[1]))
            else:
                second = np.zeros_like(conf)
                entropy = np.zeros_like(conf)
            pred = np.argmax(probs, axis=1)
            sf_conf_vals.append(conf.astype(np.float32))
            sf_margin_vals.append((conf - second).astype(np.float32))
            sf_entropy_adj_vals.append((conf * (1.0 - entropy)).astype(np.float32))
            sf_pred_hits.append((pred == tgt_np).astype(np.float32))
            for i_loc in range(len(tgt_np)):
                tg = int(tgt_np[i_loc])
                rec = per_target.setdefault(
                    tg,
                    {"conf": [], "margin": [], "entropy_adj": [], "hit": [], "centroid": [], "hf_lf": []},
                )
                rec["conf"].append(float(conf[i_loc]))
                rec["margin"].append(float(conf[i_loc] - second[i_loc]))
                rec["entropy_adj"].append(float(conf[i_loc] * (1.0 - entropy[i_loc])))
                rec["hit"].append(float(pred[i_loc] == tgt_np[i_loc]))

        fake_db_np = fake_db.detach().cpu().numpy().astype(np.float32)
        for i in range(fake_db_np.shape[0]):
            mel_db = fake_db_np[i]
            p = np.power(10.0, mel_db / 10.0).astype(np.float32)
            bin_energy = p.sum(axis=1)
            total = float(bin_energy.sum() + 1e-8)
            centroid = float((mel_freqs * bin_energy).sum() / total)
            hf = float(bin_energy[-hf_bins:].sum() / total)
            lf = float(bin_energy[:lf_bins].sum() / total)
            hf_lf = float(hf / (lf + 1e-8))
            fake_centroid_vals.append(centroid)
            fake_hf_lf_ratio_vals.append(hf_lf)
            feat = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)], axis=0)
            fake_collapse_feats.append(feat.astype(np.float32))
            tg_i = int(tgt_idx[i].item())
            rec = per_target.setdefault(
                tg_i,
                {"conf": [], "margin": [], "entropy_adj": [], "hit": [], "centroid": [], "hf_lf": []},
            )
            rec["centroid"].append(float(centroid))
            rec["hf_lf"].append(float(hf_lf))

        n += len(tgt_idx)

    if n == 0 or not sf_conf_vals:
        return {
            "n_eval": 0,
            "mps": float("nan"),
            "style_fidelity_conf": float("nan"),
            "style_acc": float("nan"),
            "spectral_continuity": float("nan"),
            "fake_centroid_hz": float("nan"),
            "fake_hf_lf_ratio": float("nan"),
            "fake_pairwise_feature_cos": float("nan"),
            "passes": {
                "mps": False,
                "style_acc": False,
                "style_fidelity": False,
                "spectral_continuity": False,
                "anti_hf_lf": False,
                "anti_centroid": False,
                "anti_collapse": False,
                "lab3_done": False,
            },
        }

    mps = float(np.concatenate(mps_vals).mean())
    sf_conf = float(np.concatenate(sf_conf_vals).mean())
    sf_margin = float(np.concatenate(sf_margin_vals).mean())
    sf_entropy_adj = float(np.concatenate(sf_entropy_adj_vals).mean())
    sf_acc = float(np.concatenate(sf_pred_hits).mean())
    continuity = float(np.mean(continuity_vals))
    fake_centroid = float(np.mean(fake_centroid_vals)) if fake_centroid_vals else float("nan")
    fake_hf_lf_ratio = float(np.mean(fake_hf_lf_ratio_vals)) if fake_hf_lf_ratio_vals else float("nan")
    fake_pairwise_feature_cos = (
        _pairwise_cos_mean(np.stack(fake_collapse_feats).astype(np.float32))
        if len(fake_collapse_feats) >= 2
        else float("nan")
    )

    passes = {
        "mps": bool(mps >= float(mps_threshold)),
        "style_acc": bool(sf_acc >= float(style_acc_threshold)),
        "style_fidelity": bool(sf_conf >= float(sf_threshold)),
        "spectral_continuity": bool(continuity <= float(continuity_max)),
        "anti_hf_lf": bool(fake_hf_lf_ratio <= float(hf_lf_ratio_max)),
        "anti_centroid": bool(float(spectral_centroid_min_hz) <= fake_centroid <= float(spectral_centroid_max_hz)),
        "anti_collapse": bool(fake_pairwise_feature_cos <= float(max_fake_pairwise_cos)),
    }
    passes["lab3_done"] = bool(all(bool(v) for v in passes.values()))

    per_target_metrics: Dict[str, Dict[str, float]] = {}
    for tg, rec in sorted(per_target.items(), key=lambda kv: kv[0]):
        key = str(idx_to_genre.get(int(tg), str(tg))) if idx_to_genre is not None else str(tg)
        n_t = int(max(len(rec["conf"]), len(rec["centroid"])))
        per_target_metrics[key] = {
            "n_eval": n_t,
            "style_fidelity_conf": float(np.mean(rec["conf"])) if rec["conf"] else float("nan"),
            "style_conf_margin": float(np.mean(rec["margin"])) if rec["margin"] else float("nan"),
            "style_conf_entropy_adjusted": float(np.mean(rec["entropy_adj"])) if rec["entropy_adj"] else float("nan"),
            "style_acc": float(np.mean(rec["hit"])) if rec["hit"] else float("nan"),
            "fake_centroid_hz": float(np.mean(rec["centroid"])) if rec["centroid"] else float("nan"),
            "fake_hf_lf_ratio": float(np.mean(rec["hf_lf"])) if rec["hf_lf"] else float("nan"),
        }

    return {
        "n_eval": int(n),
        "mps": mps,
        "style_fidelity_conf": sf_conf,
        "style_conf_margin": sf_margin,
        "style_conf_entropy_adjusted": sf_entropy_adj,
        "style_acc": sf_acc,
        "spectral_continuity": continuity,
        "fake_centroid_hz": fake_centroid,
        "fake_hf_lf_ratio": fake_hf_lf_ratio,
        "fake_pairwise_feature_cos": fake_pairwise_feature_cos,
        "per_target_metrics": per_target_metrics,
        "thresholds": {
            "mps": float(mps_threshold),
            "style_acc": float(style_acc_threshold),
            "style_fidelity_conf": float(sf_threshold),
            "spectral_continuity_max": float(continuity_max),
            "hf_lf_ratio_max": float(hf_lf_ratio_max),
            "spectral_centroid_min_hz": float(spectral_centroid_min_hz),
            "spectral_centroid_max_hz": float(spectral_centroid_max_hz),
            "fake_pairwise_feature_cos_max": float(max_fake_pairwise_cos),
        },
        "passes": passes,
    }


def evaluate_classifier_quality(
    style_classifier: LogisticRegression,
    z_style_val: np.ndarray,
    genre_idx_val: np.ndarray,
) -> Dict:
    pred = style_classifier.predict(z_style_val)
    acc = float(accuracy_score(genre_idx_val, pred))
    return {"style_classifier_val_acc": acc}
