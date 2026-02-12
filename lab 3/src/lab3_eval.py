from __future__ import annotations

from typing import Dict, Optional

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


@torch.no_grad()
def evaluate_genre_shift(
    generator: ReconstructionDecoder,
    frozen_encoder: FrozenLab1Encoder,
    val_loader: DataLoader,
    cond_bank: torch.Tensor,
    style_classifier: LogisticRegression,
    device: str,
    mps_threshold: float = 0.90,
    sf_threshold: float = 0.85,
    max_batches: Optional[int] = None,
) -> Dict:
    generator.eval()
    n_genres = int(cond_bank.shape[0])
    cond_bank = cond_bank.to(device)

    mps_vals = []
    sf_conf_vals = []
    sf_margin_vals = []
    sf_entropy_adj_vals = []
    sf_pred_hits = []
    continuity_vals = []
    n = 0

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
        n += len(tgt_np)

    if n == 0:
        return {
            "n_eval": 0,
            "mps": float("nan"),
            "style_fidelity_conf": float("nan"),
            "style_acc": float("nan"),
            "spectral_continuity": float("nan"),
            "passes": {
                "mps": False,
                "style_fidelity": False,
                "spectral_continuity": False,
                "lab3_done": False,
            },
        }

    mps = float(np.concatenate(mps_vals).mean())
    sf_conf = float(np.concatenate(sf_conf_vals).mean())
    sf_margin = float(np.concatenate(sf_margin_vals).mean())
    sf_entropy_adj = float(np.concatenate(sf_entropy_adj_vals).mean())
    sf_acc = float(np.concatenate(sf_pred_hits).mean())
    continuity = float(np.mean(continuity_vals))

    passes = {
        "mps": bool(mps >= float(mps_threshold)),
        "style_fidelity": bool(sf_conf >= float(sf_threshold)),
        # Continuity is reported as a quality score. No hard universal threshold.
        "spectral_continuity": True,
    }
    passes["lab3_done"] = bool(passes["mps"] and passes["style_fidelity"] and passes["spectral_continuity"])

    return {
        "n_eval": int(n),
        "mps": mps,
        "style_fidelity_conf": sf_conf,
        "style_conf_margin": sf_margin,
        "style_conf_entropy_adjusted": sf_entropy_adj,
        "style_acc": sf_acc,
        "spectral_continuity": continuity,
        "thresholds": {
            "mps": float(mps_threshold),
            "style_fidelity_conf": float(sf_threshold),
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
