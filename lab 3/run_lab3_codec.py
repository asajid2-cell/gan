from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from src.lab3_bridge import FrozenLab1Encoder
from src.lab3_codec_bridge import FrozenEncodec
from src.lab3_codec_data import (
    DEFAULT_MANIFESTS,
    CachedCodecDataset,
    assign_genres,
    build_codec_cache,
    load_codec_cache,
    load_manifests,
    materialize_genre_samples,
    save_codec_cache,
    stratified_group_split_indices,
    stratified_split_indices,
)
from src.lab3_data import apply_genre_schema, genre_num_sources, genre_source_table, materialize_genre_samples_balanced_sources
from src.lab3_codec_judge import (
    fit_codec_style_judge,
    freeze_judge,
    CodecStyleJudge,
    Lab1StyleProbe,
    fit_lab1_style_probe,
    freeze_probe,
    MERTStyleProbe,
    fit_mert_style_probe,
    freeze_mert_probe,
    fit_source_removal_projection,
    apply_source_removal_to_q_emb,
)
from src.lab3_mert_bridge import FrozenMERT
from src.lab3_codec_models import CodecLatentTranslator, CodecTrainWeights, MultiScaleWaveDiscriminator
from src.lab3_codec_train import (
    CodecStageTrainConfig,
    build_q_exemplar_bank,
    build_style_centroid_bank,
    build_style_exemplar_bank,
    train_codec_stage,
)
from src.lab3_sampling import resolve_next_run_name, validate_run_name

try:
    import soundfile as sf

    HAS_SF = True
except Exception:
    HAS_SF = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_out_root() -> Path:
    return _repo_root() / "saves2" / "lab3_codec_transfer"


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


def _save_json(obj: Dict, path: Path) -> None:
    def _default(v):
        if isinstance(v, Path):
            return str(v)
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_history(csv_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(csv_path, index=False)


def _load_models_from_ckpt(
    ckpt_path: Path,
    generator: CodecLatentTranslator,
    discriminator: MultiScaleWaveDiscriminator,
    device: torch.device,
) -> bool:
    if not Path(ckpt_path).exists():
        return False
    try:
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(ckpt_path), map_location=device)
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab3 codec-latent style transfer training (EnCodec + Lab1 content).")
    p.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    p.add_argument("--out-root", type=Path, default=_default_out_root())
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--strict-run-naming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-custom-run-name", action="store_true")
    p.add_argument("--resume-dir", type=Path, default=None)
    p.add_argument("--reuse-cache-dir", type=Path, default=None)

    p.add_argument("--manifests-root", type=Path, default=Path("Z:/DataSets/_lab1_manifests"))
    p.add_argument("--manifest-files", nargs="*", default=DEFAULT_MANIFESTS)
    p.add_argument("--per-genre-samples", type=int, default=600)
    p.add_argument("--genre-schema", choices=["default4", "binary_acoustic_beats"], default="default4")
    p.add_argument("--require-min-sources-per-genre", type=int, default=1)
    p.add_argument("--balance-sources-within-genre", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-is-music", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--chunks-per-track", type=int, default=2)
    p.add_argument("--chunk-sampling", choices=["uniform", "random"], default="uniform")
    p.add_argument("--min-start-sec", type=float, default=0.0)
    p.add_argument("--max-start-sec", type=float, default=None)
    p.add_argument("--split-by-track", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=328)

    p.add_argument(
        "--style-cond-source",
        choices=["lab1_zstyle", "codec_judge_embed", "lab1_probe_embed", "mert_probe_embed", "random_genre"],
        default="codec_judge_embed",
    )
    p.add_argument("--style-loss-mode", choices=["lab1_cos", "codec_judge_ce", "lab1_probe_ce", "mert_probe_ce"], default="codec_judge_ce")
    p.add_argument("--lab1-probe-epochs", type=int, default=30)
    p.add_argument("--lab1-probe-hidden", type=int, default=256)
    p.add_argument("--lab1-probe-emb-dim", type=int, default=128)
    p.add_argument("--lab1-probe-lr", type=float, default=2e-3)
    p.add_argument("--lab1-probe-patience", type=int, default=8)
    p.add_argument("--mert-model-id", type=str, default="m-a-p/MERT-v1-95M")
    p.add_argument("--mert-layer", type=int, default=-1, help="Which hidden layer to use (-1 = last)")
    p.add_argument("--mert-probe-epochs", type=int, default=30)
    p.add_argument("--mert-probe-hidden", type=int, default=256)
    p.add_argument("--mert-probe-emb-dim", type=int, default=128)
    p.add_argument("--mert-probe-lr", type=float, default=2e-3)
    p.add_argument("--mert-probe-patience", type=int, default=8)
    p.add_argument("--style-judge-mode", choices=["lab1_zstyle", "codec_judge"], default="codec_judge")
    p.add_argument("--style-judge-epochs", type=int, default=20)
    p.add_argument("--style-judge-lr", type=float, default=2e-3)
    p.add_argument("--style-judge-batch-size", type=int, default=64)
    p.add_argument("--style-judge-hidden", type=int, default=256)
    p.add_argument("--style-judge-emb-dim", type=int, default=128)
    p.add_argument("--style-judge-source-adv-weight", type=float, default=0.30)
    p.add_argument("--style-judge-source-grl-lambda", type=float, default=1.0)
    p.add_argument("--style-judge-grl-warmup-epochs", type=int, default=5)
    p.add_argument("--style-judge-patience", type=int, default=0, help="Early stopping patience; 0=disabled")
    p.add_argument("--style-judge-min-val-acc", type=float, default=0.80)
    p.add_argument("--fail-on-style-judge-weak", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--refit-style-judge", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--source-removal-projection", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--source-removal-max-frac", type=float, default=0.40,
                    help="Max fraction of latent dims to remove for source decontamination")
    p.add_argument("--source-removal-max-iters", type=int, default=20,
                    help="Max INLP iterations for source removal")

    p.add_argument("--lab1-checkpoint", type=Path, default=_default_lab1_checkpoint())
    p.add_argument("--lab1-n-frames", type=int, default=256)
    p.add_argument("--codec-model-id", type=str, default="facebook/encodec_24khz")
    p.add_argument("--codec-bandwidth", type=float, default=6.0)
    p.add_argument("--codec-chunk-seconds", type=float, default=5.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--translator-hidden-channels", type=int, default=256)
    p.add_argument("--translator-blocks", type=int, default=10)
    p.add_argument("--translator-noise-dim", type=int, default=32)
    p.add_argument("--translator-residual-scale", type=float, default=0.5)
    p.add_argument("--translator-direct-output", action="store_true",
                    help="Remove residual connection — network outputs q_hat directly")
    p.add_argument("--discriminator-scales", type=int, default=3)

    p.add_argument("--stage1-epochs", type=int, default=8)
    p.add_argument("--stage2-epochs", type=int, default=16)
    p.add_argument("--stage3-epochs", type=int, default=8)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")

    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--stage2-d-lr-mult", type=float, default=0.2)
    p.add_argument("--stage3-d-lr-mult", type=float, default=0.2)
    p.add_argument("--adv-weight", type=float, default=0.4)
    p.add_argument("--feature-match-weight", type=float, default=1.0)
    p.add_argument("--latent-l1-weight", type=float, default=4.0)
    p.add_argument("--continuity-weight", type=float, default=1.0)
    p.add_argument("--content-weight", type=float, default=2.0)
    p.add_argument("--style-weight", type=float, default=3.0)
    p.add_argument("--mrstft-weight", type=float, default=2.0)
    p.add_argument("--mode-seeking-weight", type=float, default=1.0)
    p.add_argument("--style-push-margin", type=float, default=0.30)
    p.add_argument("--stage2-style-push-weight", type=float, default=1.0)
    p.add_argument("--stage3-style-push-weight", type=float, default=1.5)
    p.add_argument("--stage2-delta-budget", type=float, default=0.12)
    p.add_argument("--stage3-delta-budget", type=float, default=0.12)
    p.add_argument("--stage2-delta-budget-weight", type=float, default=1.0)
    p.add_argument("--stage3-delta-budget-weight", type=float, default=1.0)

    p.add_argument("--stage1-adv-weight", type=float, default=0.0)
    p.add_argument("--stage1-style-weight", type=float, default=0.5)
    p.add_argument("--stage1-content-weight", type=float, default=1.0)
    p.add_argument("--stage1-mrstft-weight", type=float, default=2.0)
    p.add_argument("--stage1-latent-l1-weight", type=float, default=6.0)
    p.add_argument("--stage1-style-embed-align-weight", type=float, default=0.0)

    p.add_argument("--stage2-cond-mode", choices=["centroid", "exemplar", "mix"], default="exemplar")
    p.add_argument("--stage2-cond-alpha-start", type=float, default=0.8)
    p.add_argument("--stage2-cond-alpha-end", type=float, default=0.4)
    p.add_argument("--stage2-target-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage2-style-dropout-p", type=float, default=0.0)
    p.add_argument("--stage2-style-jitter-std", type=float, default=0.03)
    p.add_argument("--stage2-exemplar-noise-std", type=float, default=0.03)
    p.add_argument("--stage2-adv-weight", type=float, default=0.55)
    p.add_argument("--stage2-latent-l1-weight", type=float, default=0.20)
    p.add_argument("--stage2-content-weight", type=float, default=2.5)
    p.add_argument("--stage2-style-weight", type=float, default=9.0)
    p.add_argument("--stage2-mrstft-weight", type=float, default=0.20)
    p.add_argument("--stage2-style-embed-align-weight", type=float, default=0.50)

    p.add_argument("--stage3-cond-mode", choices=["centroid", "exemplar", "mix"], default="exemplar")
    p.add_argument("--stage3-cond-alpha-start", type=float, default=0.5)
    p.add_argument("--stage3-cond-alpha-end", type=float, default=0.2)
    p.add_argument("--stage3-target-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage3-style-dropout-p", type=float, default=0.25)
    p.add_argument("--stage3-style-jitter-std", type=float, default=0.08)
    p.add_argument("--stage3-exemplar-noise-std", type=float, default=0.05)
    p.add_argument("--stage3-adv-weight", type=float, default=0.70)
    p.add_argument("--stage3-latent-l1-weight", type=float, default=0.05)
    p.add_argument("--stage3-content-weight", type=float, default=2.0)
    p.add_argument("--stage3-style-weight", type=float, default=11.0)
    p.add_argument("--stage3-mrstft-weight", type=float, default=0.05)
    p.add_argument("--stage3-style-embed-align-weight", type=float, default=0.75)
    p.add_argument("--stage3-mode-seeking-weight", type=float, default=0.05)
    p.add_argument("--stage3-mode-seeking-target", type=float, default=0.03)

    p.add_argument("--r1-gamma", type=float, default=1.0)
    p.add_argument("--r1-interval", type=int, default=16)
    p.add_argument("--stage1-d-step-period", type=int, default=1)
    p.add_argument("--stage2-d-step-period", type=int, default=2)
    p.add_argument("--stage3-d-step-period", type=int, default=2)
    p.add_argument("--g-grad-clip-norm", type=float, default=5.0)
    p.add_argument("--d-grad-clip-norm", type=float, default=5.0)

    p.add_argument("--auto-sample-export", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-count", type=int, default=24)
    p.add_argument("--sample-export-tag", type=str, default="posttrain_samples")
    p.add_argument("--sample-write-source-audio", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate-multi-pass", type=int, default=1,
                    help="Run translator N times, feeding q_hat back as q_src each pass")
    p.add_argument("--gate-max-eval-samples", type=int, default=256)
    p.add_argument("--gate-min-mps", type=float, default=0.90)
    p.add_argument("--gate-min-style-conf", type=float, default=0.40)
    p.add_argument("--gate-min-style-acc", type=float, default=0.40)
    p.add_argument("--gate-max-pairwise-cos", type=float, default=0.95)
    p.add_argument("--fail-on-gate-miss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--style-bank-max-centroid-cos", type=float, default=0.98)
    p.add_argument("--style-bank-min-nearest-centroid-acc", type=float, default=0.70)
    p.add_argument("--fail-on-style-bank-collapse", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _export_codec_samples(
    out_dir: Path,
    generator: CodecLatentTranslator,
    codec: FrozenEncodec,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    val_idx: np.ndarray,
    style_centroid_bank: torch.Tensor,
    style_exemplar_bank: Optional[Dict[int, torch.Tensor]],
    seed: int = 328,
    n_samples: int = 24,
    cond_mode: str = "mix",
    cond_alpha: float = 0.35,
    write_source_audio: bool = True,
) -> Optional[Path]:
    if not HAS_SF:
        print("[sample-export] soundfile not installed; skipping export.")
        return None
    if len(val_idx) == 0:
        return None
    sample_dir = Path(out_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    n_genres = int(style_centroid_bank.shape[0])
    rows: List[Dict] = []
    device = next(generator.parameters()).device
    generator.eval()
    for i in range(int(n_samples)):
        ridx = int(val_idx[int(rng.integers(0, len(val_idx)))])
        src_genre = int(arrays["genre_idx"][ridx])
        tgt_genre = int(rng.integers(0, n_genres))
        if tgt_genre == src_genre and n_genres > 1:
            tgt_genre = int((tgt_genre + 1) % n_genres)
        q_src = torch.from_numpy(arrays["q_emb"][ridx : ridx + 1]).to(device).float()
        zc = torch.from_numpy(arrays["z_content"][ridx : ridx + 1]).to(device).float()

        z_cent = style_centroid_bank[tgt_genre : tgt_genre + 1].to(device).float()
        z_ex_bank = None if style_exemplar_bank is None else style_exemplar_bank.get(int(tgt_genre))
        if z_ex_bank is None or len(z_ex_bank) == 0:
            z_ex = z_cent
        else:
            ex_i = int(rng.integers(0, int(z_ex_bank.shape[0])))
            z_ex = z_ex_bank[ex_i : ex_i + 1].to(device).float()
        if str(cond_mode) == "centroid":
            z_tgt = z_cent
        elif str(cond_mode) == "exemplar":
            z_tgt = z_ex
        else:
            z_tgt = float(cond_alpha) * z_cent + (1.0 - float(cond_alpha)) * z_ex
        z_tgt = torch.nn.functional.normalize(z_tgt, dim=-1)

        with torch.no_grad():
            q_hat = generator(q_src=q_src, z_content=zc, z_style_tgt=z_tgt)
            wav = codec.decode_embeddings(q_hat)[0, 0].detach().cpu().numpy().astype(np.float32)
            wav = wav / (np.max(np.abs(wav)) + 1e-8)
        out_wav = sample_dir / f"sample_{i:04d}_src{src_genre}_tgt{tgt_genre}.wav"
        sf.write(str(out_wav), wav, int(codec.cfg.sample_rate))
        source_wav = sample_dir / f"sample_{i:04d}_source.wav"
        source_wav_str = ""
        source_path = ""
        source_start_sec = 0.0
        if bool(write_source_audio):
            meta = index_df.iloc[ridx]
            source_path = str(meta.get("path", ""))
            source_start_sec = float(meta.get("start_sec", 0.0))
            if source_path:
                y_src, _ = librosa.load(
                    source_path,
                    sr=int(codec.cfg.sample_rate),
                    mono=True,
                    offset=max(0.0, source_start_sec),
                    duration=float(codec.cfg.chunk_seconds),
                )
                target_len = int(round(float(codec.cfg.chunk_seconds) * float(codec.cfg.sample_rate)))
                if len(y_src) < target_len:
                    y_src = np.pad(y_src, (0, target_len - len(y_src)), mode="constant")
                elif len(y_src) > target_len:
                    y_src = y_src[:target_len]
                if np.max(np.abs(y_src)) > 0:
                    y_src = y_src / (np.max(np.abs(y_src)) + 1e-8)
                sf.write(str(source_wav), y_src.astype(np.float32), int(codec.cfg.sample_rate))
                source_wav_str = str(source_wav)
        rows.append(
            {
                "sample_id": int(i),
                "cache_row": int(ridx),
                "source_genre_idx": int(src_genre),
                "target_genre_idx": int(tgt_genre),
                "source_wav": source_wav_str,
                "source_path": source_path,
                "source_start_sec": float(source_start_sec),
                "fake_wav": str(out_wav),
            }
        )
    pd.DataFrame(rows).to_csv(sample_dir / "generation_summary.csv", index=False)
    return sample_dir


def _pairwise_cos_mean(feats: np.ndarray) -> float:
    if feats is None or len(feats) < 2:
        return float("nan")
    x = feats.astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def _style_bank_diagnostics(z_style: np.ndarray, genre_idx: np.ndarray, n_genres: int) -> Dict[str, float]:
    if n_genres < 2 or len(z_style) == 0:
        return {
            "offdiag_centroid_cos_mean": float("nan"),
            "nearest_centroid_acc": float("nan"),
        }
    z = z_style.astype(np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    centroids: List[np.ndarray] = []
    for g in range(int(n_genres)):
        m = z[genre_idx == g].mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-8)
        centroids.append(m)
    c = np.stack(centroids, axis=0)
    sim = c @ c.T
    offdiag = sim[~np.eye(sim.shape[0], dtype=bool)]
    pred = np.argmax(z @ c.T, axis=1)
    acc = float(np.mean(pred == genre_idx))
    return {
        "offdiag_centroid_cos_mean": float(np.mean(offdiag)) if offdiag.size > 0 else float("nan"),
        "nearest_centroid_acc": float(acc),
    }


@torch.no_grad()
def _build_style_bank_from_lab1_probe(
    probe: Lab1StyleProbe,
    z_style: np.ndarray,
    genre_idx: np.ndarray,
    n_genres: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[str, float]]:
    probe.eval()
    x = torch.from_numpy(z_style.astype(np.float32))
    out: List[np.ndarray] = []
    for i in range(0, int(x.shape[0]), int(max(1, batch_size))):
        xb = x[i : i + int(max(1, batch_size))].to(device)
        emb = probe.embed(xb).detach().cpu().numpy().astype(np.float32)
        out.append(emb)
    emb_all = np.concatenate(out, axis=0) if out else np.zeros((0, probe.emb_dim), dtype=np.float32)
    emb_all = emb_all / (np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-8)
    cent = build_style_centroid_bank(emb_all, genre_idx, n_genres=n_genres).to(device)
    ex = build_style_exemplar_bank(emb_all, genre_idx, n_genres=n_genres)
    diag = _style_bank_diagnostics(emb_all, genre_idx, n_genres=n_genres)
    return cent, ex, diag


@torch.no_grad()
def _build_style_bank_from_mert_probe(
    probe: MERTStyleProbe,
    mert_feat: np.ndarray,
    genre_idx: np.ndarray,
    n_genres: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[str, float]]:
    probe.eval()
    x = torch.from_numpy(mert_feat.astype(np.float32))
    out: List[np.ndarray] = []
    for i in range(0, int(x.shape[0]), int(max(1, batch_size))):
        xb = x[i : i + int(max(1, batch_size))].to(device)
        emb = probe.embed(xb).detach().cpu().numpy().astype(np.float32)
        out.append(emb)
    emb_all = np.concatenate(out, axis=0) if out else np.zeros((0, probe.emb_dim), dtype=np.float32)
    emb_all = emb_all / (np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-8)
    cent = build_style_centroid_bank(emb_all, genre_idx, n_genres=n_genres).to(device)
    ex = build_style_exemplar_bank(emb_all, genre_idx, n_genres=n_genres)
    diag = _style_bank_diagnostics(emb_all, genre_idx, n_genres=n_genres)
    return cent, ex, diag


@torch.no_grad()
def _build_style_bank_from_codec_judge(
    style_judge: CodecStyleJudge,
    q_emb: np.ndarray,
    genre_idx: np.ndarray,
    n_genres: int,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[str, float]]:
    style_judge.eval()
    x = torch.from_numpy(q_emb.astype(np.float32))
    out: List[np.ndarray] = []
    for i in range(0, int(x.shape[0]), int(max(1, batch_size))):
        xb = x[i : i + int(max(1, batch_size))].to(device)
        emb = style_judge.embed(xb).detach().cpu().numpy().astype(np.float32)
        out.append(emb)
    emb_all = np.concatenate(out, axis=0) if out else np.zeros((0, style_judge.emb_dim), dtype=np.float32)
    if emb_all.shape[0] != q_emb.shape[0]:
        raise RuntimeError("Codec judge embedding extraction failed; shape mismatch.")
    emb_all = emb_all / (np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-8)
    cent = build_style_centroid_bank(emb_all, genre_idx, n_genres=n_genres).to(device)
    ex = build_style_exemplar_bank(emb_all, genre_idx, n_genres=n_genres)
    diag = _style_bank_diagnostics(emb_all, genre_idx, n_genres=n_genres)
    return cent, ex, diag


def _evaluate_codec_gate(
    generator: CodecLatentTranslator,
    codec: FrozenEncodec,
    lab1: FrozenLab1Encoder,
    style_judge: Optional[CodecStyleJudge],
    arrays: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    style_centroid_bank: torch.Tensor,
    style_exemplar_bank: Optional[Dict[int, torch.Tensor]],
    cond_mode: str,
    cond_alpha: float,
    device: torch.device,
    max_eval_samples: int,
    seed: int,
    n_passes: int = 1,
) -> Dict[str, float]:
    n_genres = int(style_centroid_bank.shape[0])
    if n_genres < 2 or len(val_idx) == 0:
        return {
            "n_eval": 0,
            "mps": float("nan"),
            "style_conf": float("nan"),
            "style_acc": float("nan"),
            "pairwise_cos": float("nan"),
        }

    clf = None
    if style_judge is None:
        clf = LogisticRegression(max_iter=1200, random_state=int(seed), n_jobs=1)
        clf.fit(arrays["z_style"][train_idx], arrays["genre_idx"][train_idx])

    generator.eval()
    rng = np.random.default_rng(int(seed) + 17)
    eval_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(eval_idx)
    eval_idx = eval_idx[: int(min(len(eval_idx), int(max_eval_samples)))]

    codec_sr = int(codec.cfg.sample_rate)
    lab1_sr = int(lab1.cfg.sample_rate)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(lab1_sr),
        n_fft=1024,
        hop_length=256,
        n_mels=96,
        f_min=20.0,
        f_max=float(lab1_sr) * 0.5,
        power=2.0,
        center=True,
    ).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    def _wav_to_lab1_mel(wav_b_1_t: torch.Tensor) -> torch.Tensor:
        y = wav_b_1_t.squeeze(1)
        if codec_sr != lab1_sr:
            y = torchaudio.functional.resample(
                waveform=y,
                orig_freq=int(codec_sr),
                new_freq=int(lab1_sr),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_hann",
            )
        mel_pow = mel(y).clamp_min(1e-8)
        mel_db = to_db(mel_pow)
        t = int(mel_db.shape[-1])
        n_frames = 256
        if t > n_frames:
            start = (t - n_frames) // 2
            mel_db = mel_db[:, :, start : start + n_frames]
        elif t < n_frames:
            pad = torch.full(
                (mel_db.shape[0], mel_db.shape[1], n_frames - t),
                fill_value=-80.0,
                device=mel_db.device,
                dtype=mel_db.dtype,
            )
            mel_db = torch.cat([mel_db, pad], dim=2)
        return mel_db

    mps_vals: List[float] = []
    style_conf_vals: List[float] = []
    style_hit_vals: List[float] = []
    feat_all: List[np.ndarray] = []

    bs = 8
    for start in range(0, len(eval_idx), bs):
        ridx = eval_idx[start : start + bs]
        if len(ridx) == 0:
            continue
        src = arrays["genre_idx"][ridx].astype(np.int64)
        tgt = (src + 1 + (np.arange(len(src), dtype=np.int64) % max(1, n_genres - 1))) % n_genres
        clash = tgt == src
        tgt[clash] = (tgt[clash] + 1) % n_genres

        q_src = torch.from_numpy(arrays["q_emb"][ridx]).to(device).float()
        zc_src = torch.from_numpy(arrays["z_content"][ridx]).to(device).float()
        tgt_t = torch.from_numpy(tgt).to(device).long()
        z_cent = style_centroid_bank[tgt_t].to(device).float()

        z_ex_rows: List[torch.Tensor] = []
        for g in tgt.tolist():
            bank = None if style_exemplar_bank is None else style_exemplar_bank.get(int(g))
            if bank is None or int(bank.shape[0]) == 0:
                z_ex_rows.append(z_cent[len(z_ex_rows) : len(z_ex_rows) + 1])
            else:
                j = int(rng.integers(0, int(bank.shape[0])))
                z_ex_rows.append(bank[j : j + 1].to(device).float())
        z_ex = torch.cat(z_ex_rows, dim=0)
        mode = str(cond_mode).strip().lower()
        if mode == "centroid":
            z_tgt = z_cent
        elif mode == "exemplar":
            z_tgt = z_ex
        else:
            z_tgt = float(cond_alpha) * z_cent + (1.0 - float(cond_alpha)) * z_ex
        z_tgt = torch.nn.functional.normalize(z_tgt, dim=-1)

        with torch.no_grad():
            z_noise = torch.zeros((q_src.shape[0], generator.noise_dim), device=device, dtype=q_src.dtype)
            q_cur = q_src
            for _pass in range(max(1, int(n_passes))):
                q_cur = generator(q_src=q_cur, z_content=zc_src, z_style_tgt=z_tgt, noise=z_noise)
            q_hat = q_cur
            x_hat = codec.decode_embeddings(q_hat)
            mel_hat = _wav_to_lab1_mel(x_hat)
            out_hat = lab1.forward_log_mel_tensor(mel_hat)
            zc_hat = torch.nn.functional.normalize(out_hat["z_content"], dim=-1).detach().cpu().numpy()
            if style_judge is not None:
                logits = style_judge(q_hat)
                probs_t = torch.softmax(logits, dim=1).detach().cpu().numpy()
                emb = style_judge.embed(q_hat).detach().cpu().numpy().astype(np.float32)
            else:
                zs_hat = torch.nn.functional.normalize(out_hat["z_style"], dim=-1).detach().cpu().numpy()
                probs_t = clf.predict_proba(zs_hat) if clf is not None else None
                emb = zs_hat.astype(np.float32)

        zc_src_np = arrays["z_content"][ridx]
        zc_hat_n = zc_hat / (np.linalg.norm(zc_hat, axis=1, keepdims=True) + 1e-8)
        zc_src_n = zc_src_np / (np.linalg.norm(zc_src_np, axis=1, keepdims=True) + 1e-8)
        mps_vals.extend(np.sum(zc_hat_n * zc_src_n, axis=1).tolist())

        if probs_t is not None:
            style_conf_vals.extend(probs_t[np.arange(len(tgt)), tgt].tolist())
            pred = np.argmax(probs_t, axis=1)
        else:
            pred = np.zeros((len(tgt),), dtype=np.int64)
        style_hit_vals.extend((pred == tgt).astype(np.float32).tolist())
        feat_all.append(emb)

    feat_cat = np.concatenate(feat_all, axis=0) if feat_all else np.zeros((0, arrays["z_style"].shape[1]), dtype=np.float32)
    return {
        "n_eval": int(len(mps_vals)),
        "mps": float(np.mean(mps_vals)) if mps_vals else float("nan"),
        "style_conf": float(np.mean(style_conf_vals)) if style_conf_vals else float("nan"),
        "style_acc": float(np.mean(style_hit_vals)) if style_hit_vals else float("nan"),
        "pairwise_cos": _pairwise_cos_mean(feat_cat),
    }


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.per_genre_samples = min(int(args.per_genre_samples), 48)
        args.chunks_per_track = 1
        args.batch_size = min(int(args.batch_size), 6)
        args.style_judge_epochs = min(int(args.style_judge_epochs), 2)
        args.style_judge_batch_size = min(int(args.style_judge_batch_size), 32)
        args.stage1_epochs = min(int(args.stage1_epochs), 1)
        args.stage2_epochs = min(int(args.stage2_epochs), 1)
        args.stage3_epochs = min(int(args.stage3_epochs), 1)
        args.max_batches_per_epoch = 2
        args.sample_count = min(int(args.sample_count), 8)
        args.mert_probe_epochs = min(int(args.mert_probe_epochs), 2)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "fresh":
        run_name = validate_run_name(
            run_name=args.run_name.strip() if args.run_name else resolve_next_run_name(out_root),
            strict_run_naming=bool(args.strict_run_naming),
            force_custom_run_name=bool(args.force_custom_run_name),
        )
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.resume_dir is None:
            raise ValueError("--mode resume requires --resume-dir")
        out_dir = Path(args.resume_dir)
        run_name = out_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "run_state.json"
    history_path = out_dir / "history.csv"
    cache_dir = out_dir / "cache"
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state = _load_json(state_path) if args.mode == "resume" else {}
    if not state:
        state = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_name": run_name,
            "out_dir": str(out_dir),
            "current_stage": "init",
            "stage1_done": False,
            "stage2_done": False,
            "stage3_done": False,
        }
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    state["config"] = vars(args)
    _save_json(state, state_path)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else str(args.device)
    device = torch.device(device)
    print(f"[setup] run={run_name} out={out_dir}")
    print(f"[setup] device={device}")

    lab1 = FrozenLab1Encoder(checkpoint_path=Path(args.lab1_checkpoint), device=str(device))
    codec = FrozenEncodec(
        model_id=str(args.codec_model_id),
        bandwidth=float(args.codec_bandwidth),
        chunk_seconds=float(args.codec_chunk_seconds),
        device=str(device),
    )

    # Load MERT if needed for conditioning
    need_mert = (
        str(args.style_cond_source).strip().lower() == "mert_probe_embed"
        or str(args.style_loss_mode).strip().lower() == "mert_probe_ce"
    )
    mert: Optional[FrozenMERT] = None
    if need_mert:
        mert = FrozenMERT(
            model_id=str(args.mert_model_id),
            chunk_seconds=float(args.codec_chunk_seconds),
            device=str(device),
            layer=int(args.mert_layer),
        )
        print(f"[setup] mert={args.mert_model_id} hidden={mert.cfg.hidden_size} sr={mert.cfg.sample_rate}")

    if cache_dir.exists() and (cache_dir / "codec_cache_index.csv").exists():
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=cache_dir)
    elif args.reuse_cache_dir is not None:
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=Path(args.reuse_cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
    else:
        manifests_df = load_manifests(Path(args.manifests_root), manifest_files=args.manifest_files)
        assigned_df = apply_genre_schema(assign_genres(manifests_df), schema=str(args.genre_schema))
        if bool(args.balance_sources_within_genre):
            samples_df = materialize_genre_samples_balanced_sources(
                assigned_df=assigned_df,
                per_genre_samples=int(args.per_genre_samples),
                seed=int(args.seed),
                drop_unassigned=True,
                require_existing_paths=True,
                require_is_music=bool(args.require_is_music),
            )
        else:
            samples_df = materialize_genre_samples(
                assigned_df=assigned_df,
            per_genre_samples=int(args.per_genre_samples),
            seed=int(args.seed),
            drop_unassigned=True,
            require_existing_paths=True,
            require_is_music=bool(args.require_is_music),
        )
        # Ingestion audit: each genre must span multiple sources to avoid trivial source leakage.
        g_sources = genre_num_sources(samples_df)
        tbl = genre_source_table(samples_df)
        tbl.to_csv(out_dir / "ingestion_genre_source_table.csv", index=False)
        min_sources = int(args.require_min_sources_per_genre)
        if min_sources > 1:
            bad = {g: n for g, n in g_sources.items() if int(n) < min_sources}
            if bad:
                raise RuntimeError(
                    f"Genres with < {min_sources} sources: {bad}. "
                    f"See {out_dir / 'ingestion_genre_source_table.csv'} or use --genre-schema binary_acoustic_beats / --balance-sources-within-genre."
                )
        index_df, arrays, genre_to_idx, cache_meta = build_codec_cache(
            samples_df=samples_df,
            codec=codec,
            lab1_encoder=lab1,
            cache_dir=cache_dir,
            lab1_n_frames=int(args.lab1_n_frames),
            chunks_per_track=int(args.chunks_per_track),
            chunk_sampling=str(args.chunk_sampling),
            min_start_sec=float(args.min_start_sec),
            max_start_sec=args.max_start_sec,
            seed=int(args.seed),
            progress_every=50,
            mert=mert,
        )
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)

    # If MERT is needed but cache lacks mert_feat, extract features from audio paths
    if need_mert and "mert_feat" not in arrays and mert is not None:
        print("[mert-cache] extracting MERT features for existing cache...")
        mert_feats: List[np.ndarray] = []
        for i, rec in index_df.iterrows():
            p = Path(str(rec["path"]))
            start_sec = float(rec.get("start_sec", 0.0))
            try:
                wav_raw = codec.load_codec_chunk(path=p, start_sec=start_sec)
                wav_mert = codec.resample_audio(wav_raw, sr_from=int(codec.cfg.sample_rate), sr_to=int(mert.cfg.sample_rate))
                feat = mert.extract_features(wav_mert)
                mert_feats.append(feat)
            except Exception:
                mert_feats.append(np.zeros(mert.cfg.hidden_size, dtype=np.float32))
            if (i + 1) % 100 == 0:
                print(f"[mert-cache] {i + 1}/{len(index_df)}")
        arrays["mert_feat"] = np.stack(mert_feats).astype(np.float32)
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
        print(f"[mert-cache] done, shape={arrays['mert_feat'].shape}")

    # Free MERT model from GPU — only the tiny MERTStyleProbe is needed during training
    if mert is not None:
        del mert.model
        del mert.processor
        del mert
        mert = None
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[setup] freed MERT model from GPU")

    print(f"[cache] rows={len(index_df)} genres={len(genre_to_idx)}")

    genre_idx = arrays["genre_idx"]
    if bool(args.split_by_track) and "track_id" in index_df.columns:
        train_idx, val_idx = stratified_group_split_indices(
            genre_idx=genre_idx,
            group_ids=index_df["track_id"].to_numpy(),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    else:
        train_idx, val_idx = stratified_split_indices(
            genre_idx=genre_idx,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    print(f"[split] train={len(train_idx)} val={len(val_idx)}")

    # Source factorization (needed early for optional source-removal projection)
    source_codes, source_names = pd.factorize(index_df["source"].astype(str), sort=True)
    source_idx_all = source_codes.astype(np.int64)
    n_sources = int(len(source_names))
    source_name_map = {int(i): str(name) for i, name in enumerate(source_names.tolist())}
    state["source_index_map"] = {str(k): v for k, v in source_name_map.items()}

    # Optional source-removal projection: remove source-predictive directions from q_emb
    source_removal_info: Dict[str, object] = {"enabled": False}
    if bool(args.source_removal_projection) and n_sources > 1:
        sr_result = fit_source_removal_projection(
            q_emb=arrays["q_emb"],
            source_idx=source_idx_all,
            genre_idx=arrays["genre_idx"],
            n_sources=n_sources,
            seed=int(args.seed),
            max_remove_frac=float(args.source_removal_max_frac),
            max_iterations=int(args.source_removal_max_iters),
        )
        arrays["q_emb"] = apply_source_removal_to_q_emb(arrays["q_emb"], sr_result.projection)
        source_removal_info = {
            "enabled": True,
            "n_removed_dims": sr_result.n_removed_dims,
            "max_remove_frac": float(args.source_removal_max_frac),
            "source_acc_before": sr_result.source_acc_before,
            "source_acc_after": sr_result.source_acc_after,
            "genre_acc_before": sr_result.genre_acc_before,
            "genre_acc_after": sr_result.genre_acc_after,
        }
        np.save(str(out_dir / "source_removal_projection.npy"), sr_result.projection)
    _save_json(source_removal_info, out_dir / "source_removal_info.json")
    state["source_removal"] = source_removal_info
    _save_json(state, state_path)

    n_genres = len(genre_to_idx)
    train_ds = CachedCodecDataset(arrays=arrays, indices=train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Empty training loader. Increase samples or reduce batch-size.")

    z_style_train = arrays["z_style"][train_idx]
    genre_train = arrays["genre_idx"][train_idx]
    q_train = arrays["q_emb"][train_idx]

    q_exemplar_bank = build_q_exemplar_bank(q_train, genre_train, n_genres=n_genres)
    style_judge: Optional[CodecStyleJudge] = None
    style_judge_info: Dict[str, object] = {
        "mode": str(args.style_judge_mode),
        "source_adv_weight": float(args.style_judge_source_adv_weight),
        "source_grl_lambda": float(args.style_judge_source_grl_lambda),
        "n_sources": int(n_sources),
    }
    if str(args.style_judge_mode).strip().lower() == "codec_judge":
        judge_path = out_dir / "codec_style_judge.pt"
        if judge_path.exists() and bool(args.mode == "resume") and not bool(args.refit_style_judge):
            payload = torch.load(str(judge_path), map_location="cpu")
            cfgj = payload.get("config", {})
            style_judge = CodecStyleJudge(
                in_channels=int(cfgj.get("in_channels", cache_meta.codec_channels)),
                n_genres=int(cfgj.get("n_genres", n_genres)),
                hidden=int(cfgj.get("hidden", int(args.style_judge_hidden))),
                emb_dim=int(cfgj.get("emb_dim", int(args.style_judge_emb_dim))),
                n_sources=int(cfgj.get("n_sources", 0)),
            ).to(device)
            style_judge.load_state_dict(payload["model"], strict=True)
            style_judge = freeze_judge(style_judge)
            style_judge_info.update(payload.get("metrics", {}))
            print(f"[style-judge] loaded {judge_path.name}")
        else:
            style_judge, fit_res = fit_codec_style_judge(
                arrays=arrays,
                train_idx=train_idx,
                val_idx=val_idx,
                n_genres=int(n_genres),
                source_idx=source_idx_all,
                n_sources=int(n_sources),
                device=device,
                epochs=int(args.style_judge_epochs),
                lr=float(args.style_judge_lr),
                batch_size=int(args.style_judge_batch_size),
                num_workers=int(args.num_workers),
                hidden=int(args.style_judge_hidden),
                emb_dim=int(args.style_judge_emb_dim),
                seed=int(args.seed),
                source_adv_weight=float(args.style_judge_source_adv_weight),
                source_grl_lambda=float(args.style_judge_source_grl_lambda),
                grl_warmup_epochs=int(args.style_judge_grl_warmup_epochs),
                patience=int(args.style_judge_patience),
            )
            style_judge = freeze_judge(style_judge)
            style_judge_info.update(
                {
                    "best_val_acc": float(fit_res.best_val_acc),
                    "last_val_acc": float(fit_res.last_val_acc),
                    "train_loss": float(fit_res.train_loss),
                    "best_source_val_acc": float(fit_res.best_source_val_acc),
                    "last_source_val_acc": float(fit_res.last_source_val_acc),
                    "source_train_loss": float(fit_res.source_train_loss),
                }
            )
            torch.save(
                {
                    "model": style_judge.state_dict(),
                    "config": {
                        "in_channels": int(cache_meta.codec_channels),
                        "n_genres": int(n_genres),
                        "hidden": int(args.style_judge_hidden),
                        "emb_dim": int(args.style_judge_emb_dim),
                        "n_sources": int(n_sources),
                    },
                    "metrics": dict(style_judge_info),
                },
                str(judge_path),
            )
            print(f"[style-judge] saved {judge_path.name}")

        min_acc = float(args.style_judge_min_val_acc)
        acc = float(style_judge_info.get("best_val_acc", float("nan")))
        if bool(args.fail_on_style_judge_weak) and (not np.isfinite(acc) or acc < min_acc):
            raise RuntimeError(f"Style judge too weak (val_acc={acc:.4f} < {min_acc:.4f}); aborting.")

    _save_json(style_judge_info, out_dir / "codec_style_judge_info.json")
    state["codec_style_judge"] = style_judge_info
    _save_json(state, state_path)

    lab1_style_diag = _style_bank_diagnostics(z_style_train, genre_train, n_genres=n_genres)
    _save_json({"metrics": lab1_style_diag}, out_dir / "lab1_style_bank_diagnostics.json")

    # Train Lab1 style probe if needed for conditioning or loss
    lab1_probe: Optional[Lab1StyleProbe] = None
    lab1_probe_info: Dict[str, object] = {"enabled": False}
    need_probe = (
        str(args.style_cond_source).strip().lower() == "lab1_probe_embed"
        or str(args.style_loss_mode).strip().lower() == "lab1_probe_ce"
    )
    if need_probe:
        probe_path = out_dir / "lab1_style_probe.pt"
        if probe_path.exists() and args.mode == "resume":
            payload = torch.load(str(probe_path), map_location="cpu")
            cfgp = payload.get("config", {})
            lab1_probe = Lab1StyleProbe(
                in_dim=int(cfgp.get("in_dim", arrays["z_style"].shape[1])),
                n_genres=int(cfgp.get("n_genres", n_genres)),
                hidden=int(cfgp.get("hidden", int(args.lab1_probe_hidden))),
                emb_dim=int(cfgp.get("emb_dim", int(args.lab1_probe_emb_dim))),
                n_sources=int(cfgp.get("n_sources", 0)),
            ).to(device)
            lab1_probe.load_state_dict(payload["model"], strict=True)
            lab1_probe = freeze_probe(lab1_probe)
            lab1_probe_info = {"enabled": True, **payload.get("metrics", {})}
            print(f"[lab1-probe] loaded {probe_path.name}")
        else:
            lab1_probe, probe_fit = fit_lab1_style_probe(
                z_style=arrays["z_style"],
                genre_idx=arrays["genre_idx"],
                train_idx=train_idx,
                val_idx=val_idx,
                n_genres=int(n_genres),
                source_idx=source_idx_all,
                n_sources=int(n_sources),
                device=device,
                epochs=int(args.lab1_probe_epochs),
                lr=float(args.lab1_probe_lr),
                batch_size=int(args.style_judge_batch_size),
                num_workers=int(args.num_workers),
                hidden=int(args.lab1_probe_hidden),
                emb_dim=int(args.lab1_probe_emb_dim),
                seed=int(args.seed),
                source_adv_weight=float(args.style_judge_source_adv_weight),
                source_grl_lambda=float(args.style_judge_source_grl_lambda),
                patience=int(args.lab1_probe_patience),
            )
            lab1_probe = freeze_probe(lab1_probe)
            lab1_probe_info = {
                "enabled": True,
                "best_val_acc": float(probe_fit.best_val_acc),
                "last_val_acc": float(probe_fit.last_val_acc),
                "train_loss": float(probe_fit.train_loss),
            }
            torch.save(
                {
                    "model": lab1_probe.state_dict(),
                    "config": {
                        "in_dim": int(arrays["z_style"].shape[1]),
                        "n_genres": int(n_genres),
                        "hidden": int(args.lab1_probe_hidden),
                        "emb_dim": int(args.lab1_probe_emb_dim),
                        "n_sources": int(n_sources),
                    },
                    "metrics": dict(lab1_probe_info),
                },
                str(probe_path),
            )
            print(f"[lab1-probe] saved {probe_path.name}")
    _save_json(lab1_probe_info, out_dir / "lab1_style_probe_info.json")
    state["lab1_style_probe"] = lab1_probe_info
    _save_json(state, state_path)

    # Train MERT style probe if needed for conditioning or loss
    mert_probe: Optional[MERTStyleProbe] = None
    mert_probe_info: Dict[str, object] = {"enabled": False}
    if need_mert and "mert_feat" in arrays:
        mert_probe_path = out_dir / "mert_style_probe.pt"
        if mert_probe_path.exists() and args.mode == "resume":
            payload = torch.load(str(mert_probe_path), map_location="cpu")
            cfgm = payload.get("config", {})
            mert_probe = MERTStyleProbe(
                in_dim=int(cfgm.get("in_dim", arrays["mert_feat"].shape[1])),
                n_genres=int(cfgm.get("n_genres", n_genres)),
                hidden=int(cfgm.get("hidden", int(args.mert_probe_hidden))),
                emb_dim=int(cfgm.get("emb_dim", int(args.mert_probe_emb_dim))),
                n_sources=int(cfgm.get("n_sources", 0)),
            ).to(device)
            mert_probe.load_state_dict(payload["model"], strict=True)
            mert_probe = freeze_mert_probe(mert_probe)
            mert_probe_info = {"enabled": True, **payload.get("metrics", {})}
            print(f"[mert-probe] loaded {mert_probe_path.name}")
        else:
            mert_probe, mert_fit = fit_mert_style_probe(
                mert_feat=arrays["mert_feat"],
                genre_idx=arrays["genre_idx"],
                train_idx=train_idx,
                val_idx=val_idx,
                n_genres=int(n_genres),
                source_idx=source_idx_all,
                n_sources=int(n_sources),
                device=device,
                epochs=int(args.mert_probe_epochs),
                lr=float(args.mert_probe_lr),
                batch_size=int(args.style_judge_batch_size),
                num_workers=int(args.num_workers),
                hidden=int(args.mert_probe_hidden),
                emb_dim=int(args.mert_probe_emb_dim),
                seed=int(args.seed),
                source_adv_weight=float(args.style_judge_source_adv_weight),
                source_grl_lambda=float(args.style_judge_source_grl_lambda),
                patience=int(args.mert_probe_patience),
            )
            mert_probe = freeze_mert_probe(mert_probe)
            mert_probe_info = {
                "enabled": True,
                "best_val_acc": float(mert_fit.best_val_acc),
                "last_val_acc": float(mert_fit.last_val_acc),
                "train_loss": float(mert_fit.train_loss),
            }
            torch.save(
                {
                    "model": mert_probe.state_dict(),
                    "config": {
                        "in_dim": int(arrays["mert_feat"].shape[1]),
                        "n_genres": int(n_genres),
                        "hidden": int(args.mert_probe_hidden),
                        "emb_dim": int(args.mert_probe_emb_dim),
                        "n_sources": int(n_sources),
                    },
                    "metrics": dict(mert_probe_info),
                },
                str(mert_probe_path),
            )
            print(f"[mert-probe] saved {mert_probe_path.name}")
    _save_json(mert_probe_info, out_dir / "mert_style_probe_info.json")
    state["mert_style_probe"] = mert_probe_info
    _save_json(state, state_path)

    cond_source = str(args.style_cond_source).strip().lower()
    if cond_source == "mert_probe_embed":
        if mert_probe is None or "mert_feat" not in arrays:
            raise RuntimeError("style_cond_source=mert_probe_embed requires MERT features and trained probe")
        mert_train = arrays["mert_feat"][train_idx]
        style_centroid_bank, style_exemplar_bank, style_bank_diag = _build_style_bank_from_mert_probe(
            probe=mert_probe,
            mert_feat=mert_train,
            genre_idx=genre_train,
            n_genres=n_genres,
            device=device,
        )
    elif cond_source == "lab1_probe_embed":
        if lab1_probe is None:
            raise RuntimeError("style_cond_source=lab1_probe_embed requires lab1 probe to be trained")
        style_centroid_bank, style_exemplar_bank, style_bank_diag = _build_style_bank_from_lab1_probe(
            probe=lab1_probe,
            z_style=z_style_train,
            genre_idx=genre_train,
            n_genres=n_genres,
            device=device,
        )
    elif cond_source == "lab1_zstyle":
        style_centroid_bank = build_style_centroid_bank(z_style_train, genre_train, n_genres=n_genres).to(device)
        style_exemplar_bank = build_style_exemplar_bank(z_style_train, genre_train, n_genres=n_genres)
        style_bank_diag = lab1_style_diag
    elif cond_source == "codec_judge_embed":
        if style_judge is None:
            raise RuntimeError("style_cond_source=codec_judge_embed requires --style-judge-mode codec_judge")
        style_centroid_bank, style_exemplar_bank, style_bank_diag = _build_style_bank_from_codec_judge(
            style_judge=style_judge,
            q_emb=q_train,
            genre_idx=genre_train,
            n_genres=n_genres,
            device=device,
            batch_size=max(32, int(args.batch_size) * 2),
        )
    else:
        rng_cond = np.random.default_rng(int(args.seed) + 913)
        dim = int(args.style_judge_emb_dim) if style_judge is not None else int(arrays["z_style"].shape[1])
        cond = rng_cond.standard_normal((int(n_genres), dim), dtype=np.float32)
        cond = cond / (np.linalg.norm(cond, axis=1, keepdims=True) + 1e-8)
        style_centroid_bank = torch.from_numpy(cond).to(device)
        style_exemplar_bank = None
        z_fake = np.repeat(cond, repeats=8, axis=0)
        y_fake = np.repeat(np.arange(int(n_genres), dtype=np.int64), repeats=8, axis=0)
        style_bank_diag = _style_bank_diagnostics(z_fake, y_fake, n_genres=n_genres)

    style_bank_pass = {
        "centroid_cos": bool(style_bank_diag["offdiag_centroid_cos_mean"] <= float(args.style_bank_max_centroid_cos)),
        "nearest_centroid_acc": bool(style_bank_diag["nearest_centroid_acc"] >= float(args.style_bank_min_nearest_centroid_acc)),
    }
    style_bank_pass["all"] = bool(all(style_bank_pass.values()))
    style_bank_out = {
        "source": cond_source,
        "metrics": style_bank_diag,
        "thresholds": {
            "max_centroid_cos": float(args.style_bank_max_centroid_cos),
            "min_nearest_centroid_acc": float(args.style_bank_min_nearest_centroid_acc),
        },
        "passes": style_bank_pass,
    }
    _save_json(style_bank_out, out_dir / "style_bank_diagnostics.json")
    state["style_bank_diagnostics"] = style_bank_out
    _save_json(state, state_path)
    print(
        "[style-bank]"
        f" source={cond_source}"
        f" centroid_cos={style_bank_diag['offdiag_centroid_cos_mean']:.4f}"
        f" nc_acc={style_bank_diag['nearest_centroid_acc']:.4f}"
        f" pass={style_bank_pass['all']}"
    )
    if bool(args.fail_on_style_bank_collapse) and not bool(style_bank_pass["all"]):
        raise RuntimeError("Style bank collapsed; aborting before training.")

    gen = CodecLatentTranslator(
        in_channels=int(cache_meta.codec_channels),
        z_content_dim=int(arrays["z_content"].shape[1]),
        z_style_dim=int(style_centroid_bank.shape[1]),
        hidden_channels=int(args.translator_hidden_channels),
        n_blocks=int(args.translator_blocks),
        noise_dim=int(args.translator_noise_dim),
        residual_scale=float(args.translator_residual_scale),
        direct_output=bool(args.translator_direct_output),
    ).to(device)
    disc = MultiScaleWaveDiscriminator(n_scales=int(args.discriminator_scales)).to(device)

    stage1_weights = CodecTrainWeights(
        adv=float(args.stage1_adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.stage1_latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.stage1_content_weight),
        style=float(args.stage1_style_weight),
        mrstft=float(args.stage1_mrstft_weight),
        mode_seeking=0.0,
    )
    stage2_weights = CodecTrainWeights(
        adv=float(args.stage2_adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.stage2_latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.stage2_content_weight),
        style=float(args.stage2_style_weight),
        mrstft=float(args.stage2_mrstft_weight),
        mode_seeking=0.0,
        style_push=float(args.stage2_style_push_weight),
        delta_budget=float(args.stage2_delta_budget_weight),
    )
    stage3_weights = CodecTrainWeights(
        adv=float(args.stage3_adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.stage3_latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.stage3_content_weight),
        style=float(args.stage3_style_weight),
        mrstft=float(args.stage3_mrstft_weight),
        mode_seeking=float(args.stage3_mode_seeking_weight),
        style_push=float(args.stage3_style_push_weight),
        delta_budget=float(args.stage3_delta_budget_weight),
    )

    if not args.skip_stage1 and not bool(state.get("stage1_done", False)):
        state["current_stage"] = "stage1"
        _save_json(state, state_path)
        stage1_style_mode = "lab1_cos"
        if (
            style_judge is not None
            and str(args.style_cond_source).strip().lower() == "codec_judge_embed"
        ):
            stage1_style_mode = "codec_judge_ce"
        elif lab1_probe is not None and str(args.style_loss_mode).strip().lower() == "lab1_probe_ce":
            stage1_style_mode = "lab1_probe_ce"
        elif mert_probe is not None and str(args.style_loss_mode).strip().lower() == "mert_probe_ce":
            stage1_style_mode = "mert_probe_ce"
        cfg1 = CodecStageTrainConfig(
            stage_name="stage1",
            epochs=int(args.stage1_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode="centroid",
            cond_alpha_start=1.0,
            cond_alpha_end=1.0,
            target_balance=True,
            style_dropout_p=0.0,
            style_jitter_std=0.0,
            exemplar_noise_std=0.0,
            d_step_period=int(args.stage1_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage1_weights,
            style_loss_mode=stage1_style_mode,
            style_embed_align_weight=float(args.stage1_style_embed_align_weight),
        )
        hist = train_codec_stage(
            stage_cfg=cfg1,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            style_judge=style_judge,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume"),
            lab1_probe=lab1_probe,
            mert_probe=mert_probe,
        )
        _append_history(history_path, hist)
        state["stage1_done"] = True
        _save_json(state, state_path)

    if not args.skip_stage2 and not bool(state.get("stage2_done", False)):
        stage2_ckpt = checkpoints_dir / "stage2_latest.pt"
        if args.mode == "resume" and not stage2_ckpt.exists():
            _load_models_from_ckpt(
                ckpt_path=checkpoints_dir / "stage1_latest.pt",
                generator=gen,
                discriminator=disc,
                device=device,
            )
        state["current_stage"] = "stage2"
        _save_json(state, state_path)
        cfg2 = CodecStageTrainConfig(
            stage_name="stage2",
            epochs=int(args.stage2_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d) * float(args.stage2_d_lr_mult),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode=str(args.stage2_cond_mode),
            cond_alpha_start=float(args.stage2_cond_alpha_start),
            cond_alpha_end=float(args.stage2_cond_alpha_end),
            target_balance=bool(args.stage2_target_balance),
            style_dropout_p=float(args.stage2_style_dropout_p),
            style_jitter_std=float(args.stage2_style_jitter_std),
            exemplar_noise_std=float(args.stage2_exemplar_noise_std),
            d_step_period=int(args.stage2_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage2_weights,
            style_push_margin=float(args.style_push_margin),
            delta_budget=float(args.stage2_delta_budget),
            style_loss_mode=str(args.style_loss_mode),
            style_embed_align_weight=float(args.stage2_style_embed_align_weight),
        )
        hist = train_codec_stage(
            stage_cfg=cfg2,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            style_judge=style_judge,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume" and stage2_ckpt.exists()),
            lab1_probe=lab1_probe,
            mert_probe=mert_probe,
        )
        _append_history(history_path, hist)
        state["stage2_done"] = True
        _save_json(state, state_path)

    if not args.skip_stage3 and not bool(state.get("stage3_done", False)):
        stage3_ckpt = checkpoints_dir / "stage3_latest.pt"
        if args.mode == "resume" and not stage3_ckpt.exists():
            loaded = _load_models_from_ckpt(
                ckpt_path=checkpoints_dir / "stage2_latest.pt",
                generator=gen,
                discriminator=disc,
                device=device,
            )
            if not loaded:
                _load_models_from_ckpt(
                    ckpt_path=checkpoints_dir / "stage1_latest.pt",
                    generator=gen,
                    discriminator=disc,
                    device=device,
                )
        state["current_stage"] = "stage3"
        _save_json(state, state_path)
        cfg3 = CodecStageTrainConfig(
            stage_name="stage3",
            epochs=int(args.stage3_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d) * float(args.stage3_d_lr_mult),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode=str(args.stage3_cond_mode),
            cond_alpha_start=float(args.stage3_cond_alpha_start),
            cond_alpha_end=float(args.stage3_cond_alpha_end),
            target_balance=bool(args.stage3_target_balance),
            style_dropout_p=float(args.stage3_style_dropout_p),
            style_jitter_std=float(args.stage3_style_jitter_std),
            exemplar_noise_std=float(args.stage3_exemplar_noise_std),
            d_step_period=int(args.stage3_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage3_weights,
            mode_seeking_noise_scale=1.0,
            mode_seeking_target=float(args.stage3_mode_seeking_target),
            style_push_margin=float(args.style_push_margin),
            delta_budget=float(args.stage3_delta_budget),
            style_loss_mode=str(args.style_loss_mode),
            style_embed_align_weight=float(args.stage3_style_embed_align_weight),
        )
        hist = train_codec_stage(
            stage_cfg=cfg3,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            style_judge=style_judge,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume" and stage3_ckpt.exists()),
            lab1_probe=lab1_probe,
            mert_probe=mert_probe,
        )
        _append_history(history_path, hist)
        state["stage3_done"] = True
        _save_json(state, state_path)

    # Load best checkpoint if all stages were skipped (e.g. eval-only resume)
    if bool(state.get("stage3_done")) and bool(state.get("stage2_done")) and bool(state.get("stage1_done")):
        for _ckpt_name in ("stage3_latest.pt", "stage2_latest.pt", "stage1_latest.pt"):
            _ckpt_p = checkpoints_dir / _ckpt_name
            if _ckpt_p.exists():
                _load_models_from_ckpt(
                    ckpt_path=_ckpt_p,
                    generator=gen,
                    discriminator=disc,
                    device=device,
                )
                print(f"[eval-resume] loaded {_ckpt_name}")
                break

    if bool(args.auto_sample_export):
        sample_dir = _export_codec_samples(
            out_dir=out_dir / "samples" / str(args.sample_export_tag),
            generator=gen,
            codec=codec,
            index_df=index_df,
            arrays=arrays,
            val_idx=val_idx,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            seed=int(args.seed),
            n_samples=int(args.sample_count),
            cond_mode=str(args.stage3_cond_mode),
            cond_alpha=float(args.stage3_cond_alpha_end),
            write_source_audio=bool(args.sample_write_source_audio),
        )
        if sample_dir is not None:
            print(f"[sample-export] wrote {sample_dir}")

    gate_metrics = _evaluate_codec_gate(
        generator=gen,
        codec=codec,
        lab1=lab1,
        style_judge=style_judge,
        arrays=arrays,
        train_idx=train_idx,
        val_idx=val_idx,
        style_centroid_bank=style_centroid_bank,
        style_exemplar_bank=style_exemplar_bank,
        cond_mode=str(args.stage3_cond_mode),
        cond_alpha=float(args.stage3_cond_alpha_end),
        device=device,
        max_eval_samples=int(args.gate_max_eval_samples),
        seed=int(args.seed),
        n_passes=int(args.gate_multi_pass),
    )
    gate_pass = {
        "mps": bool(gate_metrics["mps"] >= float(args.gate_min_mps)),
        "style_conf": bool(gate_metrics["style_conf"] >= float(args.gate_min_style_conf)),
        "style_acc": bool(gate_metrics["style_acc"] >= float(args.gate_min_style_acc)),
        "pairwise_cos": bool(gate_metrics["pairwise_cos"] <= float(args.gate_max_pairwise_cos)),
    }
    gate_pass["all"] = bool(all(gate_pass.values()))
    gate_out = {
        "metrics": gate_metrics,
        "thresholds": {
            "min_mps": float(args.gate_min_mps),
            "min_style_conf": float(args.gate_min_style_conf),
            "min_style_acc": float(args.gate_min_style_acc),
            "max_pairwise_cos": float(args.gate_max_pairwise_cos),
        },
        "passes": gate_pass,
    }
    _save_json(gate_out, out_dir / "codec_gate_eval.json")
    state["codec_gate_eval"] = gate_out
    _save_json(state, state_path)
    print(
        "[gate]"
        f" mps={gate_metrics['mps']:.4f}"
        f" style_conf={gate_metrics['style_conf']:.4f}"
        f" style_acc={gate_metrics['style_acc']:.4f}"
        f" pairwise_cos={gate_metrics['pairwise_cos']:.4f}"
        f" pass={gate_pass['all']}"
    )
    if bool(args.fail_on_gate_miss) and not bool(gate_pass["all"]):
        raise RuntimeError("Codec gate criteria not met.")

    state["current_stage"] = "done"
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _save_json(state, state_path)
    print(f"[done] run={run_name}")


if __name__ == "__main__":
    main()
