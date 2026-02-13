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
from src.lab3_codec_judge import fit_codec_style_judge, freeze_judge, CodecStyleJudge
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
    p.add_argument("--chunks-per-track", type=int, default=2)
    p.add_argument("--chunk-sampling", choices=["uniform", "random"], default="uniform")
    p.add_argument("--min-start-sec", type=float, default=0.0)
    p.add_argument("--max-start-sec", type=float, default=None)
    p.add_argument("--split-by-track", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=328)

    p.add_argument("--style-cond-source", choices=["lab1_zstyle", "random_genre"], default="random_genre")
    p.add_argument("--style-loss-mode", choices=["lab1_cos", "codec_judge_ce"], default="codec_judge_ce")
    p.add_argument("--style-judge-mode", choices=["lab1_zstyle", "codec_judge"], default="codec_judge")
    p.add_argument("--style-judge-epochs", type=int, default=6)
    p.add_argument("--style-judge-lr", type=float, default=2e-3)
    p.add_argument("--style-judge-batch-size", type=int, default=64)
    p.add_argument("--style-judge-hidden", type=int, default=256)
    p.add_argument("--style-judge-emb-dim", type=int, default=128)
    p.add_argument("--style-judge-min-val-acc", type=float, default=0.75)
    p.add_argument("--fail-on-style-judge-weak", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--refit-style-judge", action=argparse.BooleanOptionalAction, default=False)

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
            q_hat = generator(q_src=q_src, z_content=zc_src, z_style_tgt=z_tgt, noise=z_noise)
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

    if cache_dir.exists() and (cache_dir / "codec_cache_index.csv").exists():
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=cache_dir)
    elif args.reuse_cache_dir is not None:
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=Path(args.reuse_cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
    else:
        manifests_df = load_manifests(Path(args.manifests_root), manifest_files=args.manifest_files)
        assigned_df = assign_genres(manifests_df)
        samples_df = materialize_genre_samples(
            assigned_df=assigned_df,
            per_genre_samples=int(args.per_genre_samples),
            seed=int(args.seed),
            drop_unassigned=True,
            require_existing_paths=True,
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
        )
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
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

    n_genres = len(genre_to_idx)
    z_style_train = arrays["z_style"][train_idx]
    genre_train = arrays["genre_idx"][train_idx]
    q_train = arrays["q_emb"][train_idx]

    q_exemplar_bank = build_q_exemplar_bank(q_train, genre_train, n_genres=n_genres)
    lab1_style_diag = _style_bank_diagnostics(z_style_train, genre_train, n_genres=n_genres)
    _save_json({"metrics": lab1_style_diag}, out_dir / "lab1_style_bank_diagnostics.json")

    if str(args.style_cond_source).strip().lower() == "lab1_zstyle":
        style_centroid_bank = build_style_centroid_bank(z_style_train, genre_train, n_genres=n_genres).to(device)
        style_exemplar_bank = build_style_exemplar_bank(z_style_train, genre_train, n_genres=n_genres)
        style_bank_diag = lab1_style_diag
    else:
        rng_cond = np.random.default_rng(int(args.seed) + 913)
        dim = int(arrays["z_style"].shape[1])
        cond = rng_cond.standard_normal((int(n_genres), dim), dtype=np.float32)
        cond = cond / (np.linalg.norm(cond, axis=1, keepdims=True) + 1e-8)
        style_centroid_bank = torch.from_numpy(cond).to(device)
        style_exemplar_bank = None
        z_fake = np.repeat(cond, repeats=8, axis=0)
        y_fake = np.repeat(np.arange(int(n_genres), dtype=np.int64), repeats=8, axis=0)
        style_bank_diag = _style_bank_diagnostics(z_fake, y_fake, n_genres=n_genres)
    style_bank_pass = {
        "centroid_cos": bool(
            style_bank_diag["offdiag_centroid_cos_mean"] <= float(args.style_bank_max_centroid_cos)
        ),
        "nearest_centroid_acc": bool(
            style_bank_diag["nearest_centroid_acc"] >= float(args.style_bank_min_nearest_centroid_acc)
        ),
    }
    style_bank_pass["all"] = bool(all(style_bank_pass.values()))
    style_bank_out = {
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
        f" centroid_cos={style_bank_diag['offdiag_centroid_cos_mean']:.4f}"
        f" nc_acc={style_bank_diag['nearest_centroid_acc']:.4f}"
        f" pass={style_bank_pass['all']}"
    )
    if bool(args.fail_on_style_bank_collapse) and not bool(style_bank_pass["all"]):
        raise RuntimeError("Style bank collapsed; aborting before training.")

    style_judge: Optional[CodecStyleJudge] = None
    style_judge_info: Dict[str, object] = {"mode": str(args.style_judge_mode)}
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
                device=device,
                epochs=int(args.style_judge_epochs),
                lr=float(args.style_judge_lr),
                batch_size=int(args.style_judge_batch_size),
                num_workers=int(args.num_workers),
                hidden=int(args.style_judge_hidden),
                emb_dim=int(args.style_judge_emb_dim),
                seed=int(args.seed),
            )
            style_judge = freeze_judge(style_judge)
            style_judge_info.update(
                {
                    "best_val_acc": float(fit_res.best_val_acc),
                    "last_val_acc": float(fit_res.last_val_acc),
                    "train_loss": float(fit_res.train_loss),
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

    gen = CodecLatentTranslator(
        in_channels=int(cache_meta.codec_channels),
        z_content_dim=int(arrays["z_content"].shape[1]),
        z_style_dim=int(arrays["z_style"].shape[1]),
        hidden_channels=int(args.translator_hidden_channels),
        n_blocks=int(args.translator_blocks),
        noise_dim=int(args.translator_noise_dim),
        residual_scale=float(args.translator_residual_scale),
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
            style_loss_mode="lab1_cos",
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
            style_loss_mode=str(args.style_loss_mode) if style_judge is not None else "lab1_cos",
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
            style_loss_mode=str(args.style_loss_mode) if style_judge is not None else "lab1_cos",
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
        )
        _append_history(history_path, hist)
        state["stage3_done"] = True
        _save_json(state, state_path)

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
