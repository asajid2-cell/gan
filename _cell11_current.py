
# Implementation: resumable curriculum trainer with step/epoch checkpoints

import time
from datetime import datetime

# --------------------
# Run control variables (set these before running this cell)
# --------------------
SAVENAME = "lab1_run_grl_hn_f"  # required: run folder name under ./saves/
MODE = "fresh"               # "fresh" starts new run, "resume" loads ./saves/<SAVENAME>/latest.pt
RUN_TRAINING = False         # set True to start/continue training
RUN_UNTIL_PHASE = None        # None => run all remaining phases, or set 1/2/3 to stop after that phase
REQUIRE_CUDA = True         # True => fail fast if CUDA is not available
DEVICE_PREFERENCE = "cuda"   # "cuda" or "cpu"
EXAMPLE_MAX_PER_SOURCE = 1   # examples per source for before/after snapshot
PHASE3_NEGATIVE_CAP = 4000    # cap negatives used in phase 3
PHASE3_POS_TO_NEG_RATIO = 1.0 # positives sampled per negative in phase 3
PHASE3_INCLUDE_FSD50K_NEG = False  # keep False unless you explicitly want FSD negatives
PHASE3_HARD_NEGATIVE_ENABLE = True
PHASE3_HARD_NEGATIVE_MIN_MUSIC_PROB = 0.90
PHASE3_HARD_NEGATIVE_REPEAT = 2
PHASE3_HARD_NEGATIVE_MAX = 1000
PHASE3_HARD_NEGATIVE_CSV = None  # Optional explicit path to gate_predictions.csv
PHASE3_HARD_NEGATIVE_AUDIT_ROOT = Path.cwd() / "saves" / "lab1_run_a" / "audits"
PHASE3_KEEP_NEG_DUPLICATES = True  # Keep duplicates so hard negatives are oversampled
PHASE3_HARD_NEGATIVE_LAST_N_EPOCHS = 6  # Apply hard negatives only in the final N epochs of phase 3


def resolve_hard_negative_csv(explicit_path: str | Path | None = PHASE3_HARD_NEGATIVE_CSV) -> Path | None:
    if explicit_path is not None:
        p = Path(str(explicit_path))
        return p if p.exists() else None
    root = Path(PHASE3_HARD_NEGATIVE_AUDIT_ROOT)
    if not root.exists():
        return None
    candidates = sorted(
        root.glob("**/gate_predictions.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_hard_negative_paths(
    min_music_prob: float = PHASE3_HARD_NEGATIVE_MIN_MUSIC_PROB,
    csv_path: str | Path | None = PHASE3_HARD_NEGATIVE_CSV,
    max_items: int | None = PHASE3_HARD_NEGATIVE_MAX,
) -> set[str]:
    p = resolve_hard_negative_csv(csv_path)
    if p is None:
        return set()
    df = pd.read_csv(p)
    required = {"path", "source", "music_prob"}
    if not required.issubset(df.columns):
        return set()
    hard = df[(df["source"] == "libirspeech") & (df["music_prob"] >= float(min_music_prob))].copy()
    hard = hard[hard["path"].map(lambda x: Path(str(x)).exists())]
    if len(hard) == 0:
        return set()
    hard = hard.sort_values("music_prob", ascending=False).reset_index(drop=True)
    if max_items is not None:
        hard = hard.head(int(max_items)).reset_index(drop=True)
    return set(hard["path"].astype(str).tolist())


def build_phase3_music_guard_manifest(
    negative_cap: int = PHASE3_NEGATIVE_CAP,
    pos_to_neg_ratio: float = PHASE3_POS_TO_NEG_RATIO,
    include_fsd50k_neg: bool = PHASE3_INCLUDE_FSD50K_NEG,
    enable_hard_negatives: bool = True,
    seed: int = SEED,
) -> pd.DataFrame:
    """Phase 3 should be balanced; avoid all-negative collapse in music head."""
    neg_pool = phase3_manifest.copy()

    keep_sources = ["libirspeech"]
    if include_fsd50k_neg:
        keep_sources.append("fsd50k")

    neg_base = neg_pool[neg_pool["source"].isin(keep_sources)].copy()
    if len(neg_base) == 0:
        raise FileNotFoundError("No negative sources available for phase 3")

    if negative_cap is not None and len(neg_base) > int(negative_cap):
        neg_base = neg_base.sample(int(negative_cap), random_state=seed).reset_index(drop=True)

    neg = neg_base.copy()
    if PHASE3_HARD_NEGATIVE_ENABLE and enable_hard_negatives:
        hard_paths = load_hard_negative_paths(
            min_music_prob=PHASE3_HARD_NEGATIVE_MIN_MUSIC_PROB,
            csv_path=PHASE3_HARD_NEGATIVE_CSV,
            max_items=PHASE3_HARD_NEGATIVE_MAX,
        )
        if len(hard_paths) > 0:
            hard = neg_pool[(neg_pool["source"] == "libirspeech") & (neg_pool["path"].astype(str).isin(hard_paths))].copy()
            hard = hard[hard["path"].map(lambda x: Path(str(x)).exists())].reset_index(drop=True)
            if len(hard) > 0:
                rep = max(1, int(PHASE3_HARD_NEGATIVE_REPEAT))
                hard_rep = pd.concat([hard] * rep, ignore_index=True)
                neg = pd.concat([neg_base, hard_rep], ignore_index=True)
                print(
                    f"[phase3] hard negatives added: base={len(neg_base)} hard_unique={len(hard)} "
                    f"repeat={rep} total_neg={len(neg)}"
                )

    neg["is_music"] = 0

    pos_pool = phase2_manifest.copy()
    if len(pos_pool) == 0:
        raise FileNotFoundError("No positive music pool available (phase2_manifest empty)")

    n_pos = max(1, int(len(neg) * float(pos_to_neg_ratio)))
    pos = pos_pool.sample(min(n_pos, len(pos_pool)), random_state=seed).reset_index(drop=True)
    pos["is_music"] = 1

    out = pd.concat([pos, neg], ignore_index=True)
    out = out[[c for c in ["source", "path", "ext", "size_bytes", "is_music"] if c in out.columns]]
    if not PHASE3_KEEP_NEG_DUPLICATES:
        out = out.drop_duplicates(subset=["path", "is_music"]).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def get_phase_manifest(
    phase: int,
    cfg: dict | None = None,
    phase3_enable_hard_negatives: bool | None = None,
) -> pd.DataFrame:
    if phase == 1:
        phase1_path = MANIFEST_FILES["phase1_audio"]
        if not phase1_path.exists():
            raise FileNotFoundError(
                f"Missing {phase1_path}. Render PDMX/TheSession audio first and create this manifest."
            )
        df = pd.read_csv(phase1_path)
        if "source" not in df.columns:
            df["source"] = "phase1_symbolic"
        df["is_music"] = 1
        return df[[c for c in ["source", "path", "ext", "size_bytes", "is_music"] if c in df.columns]].reset_index(drop=True)

    if phase == 2:
        return phase2_manifest[["source", "path", "ext", "size_bytes", "is_music"]].reset_index(drop=True)

    if phase == 3:
        if phase3_enable_hard_negatives is None:
            phase3_enable_hard_negatives = bool(cfg.get("phase3_enable_hard_negatives", False)) if cfg else False
        return build_phase3_music_guard_manifest(enable_hard_negatives=bool(phase3_enable_hard_negatives))

    raise ValueError("phase must be one of {1, 2, 3}")


def split_manifest_by_path(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = SEED):
    unique_paths = df[["path"]].drop_duplicates().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(unique_paths) * val_ratio)) if len(unique_paths) > 1 else 0

    val_paths = set(unique_paths.iloc[:n_val]["path"].tolist())
    train_df = df[~df["path"].isin(val_paths)].reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)

    if len(train_df) == 0 and len(val_df) > 0:
        train_df = val_df.copy()
    if len(val_df) == 0 and len(train_df) > 1:
        val_df = train_df.sample(min(len(train_df), 32), random_state=seed).reset_index(drop=True)

    return train_df, val_df


def build_global_source_map(phases):
    all_sources = []
    for phase in phases:
        try:
            d = get_phase_manifest(phase, cfg=None, phase3_enable_hard_negatives=False)
        except FileNotFoundError:
            continue
        all_sources.extend(d["source"].dropna().unique().tolist())
    all_sources = sorted(set(all_sources))
    return {s: i for i, s in enumerate(all_sources)}


class ChunkEncoder(nn.Module):
    def __init__(self, n_sources: int, z_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Linear(128, 256)
        self.content_head = nn.Linear(256, z_dim)
        self.style_head = nn.Linear(256, z_dim)
        self.style_cls = nn.Linear(z_dim, n_sources)
        self.content_style_adv = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, n_sources),
        )
        self.music_head = nn.Linear(256, 1)

    def forward(self, log_mel: torch.Tensor, grl_lambda: float = 1.0):
        x = log_mel.unsqueeze(1)
        h = self.backbone(x).flatten(1)
        h = F.relu(self.shared(h))
        z_content = F.normalize(self.content_head(h), dim=-1)
        z_style = F.normalize(self.style_head(h), dim=-1)
        z_content_rev = grad_reverse(z_content, lambda_=grl_lambda)
        return {
            "z_content": z_content,
            "z_style": z_style,
            "style_logits": self.style_cls(z_style),
            "content_style_logits": self.content_style_adv(z_content_rev),
            "music_logit": self.music_head(h).squeeze(-1),
        }


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


def compute_losses(
    out_a,
    out_b,
    source_idx,
    is_music,
    weights,
    teacher_out_a=None,
    teacher_out_b=None,
):
    loss_content = F.mse_loss(out_a["z_content"], out_b["z_content"])
    loss_style = F.cross_entropy(out_a["style_logits"], source_idx)
    loss_content_adv = F.cross_entropy(out_a["content_style_logits"], source_idx)
    loss_content_l1 = 0.5 * (
        out_a["z_content"].abs().mean() + out_b["z_content"].abs().mean()
    )

    # In all-positive/all-negative batches, music BCE can bias the gate.
    # Optionally skip these updates and train the gate only on mixed batches.
    n_pos_raw = is_music.sum()
    n_neg_raw = (1.0 - is_music).sum()
    has_both_classes = bool((n_pos_raw > 0).item() and (n_neg_raw > 0).item())
    music_only_when_mixed = bool(weights.get("music_only_when_mixed", False))
    skip_music = music_only_when_mixed and (not has_both_classes)

    if skip_music:
        loss_music = out_a["music_logit"].sum() * 0.0
        pos_weight = torch.tensor(1.0, device=is_music.device, dtype=is_music.dtype)
    else:
        n_pos = torch.clamp(n_pos_raw, min=1.0)
        n_neg = torch.clamp(n_neg_raw, min=1.0)
        if "music_pos_weight" in weights:
            pw = float(weights["music_pos_weight"])
            pos_weight = torch.tensor(pw, device=is_music.device, dtype=is_music.dtype)
        else:
            pos_weight = torch.clamp(n_neg / n_pos, min=0.25, max=8.0).detach()

        loss_music = F.binary_cross_entropy_with_logits(
            out_a["music_logit"],
            is_music,
            pos_weight=pos_weight,
        )

    loss_music_bias = out_a["music_logit"].mean().abs()
    loss_anchor = out_a["z_content"].sum() * 0.0
    if teacher_out_a is not None:
        loss_anchor = loss_anchor + F.mse_loss(out_a["z_content"], teacher_out_a["z_content"])
    if teacher_out_b is not None:
        loss_anchor = loss_anchor + F.mse_loss(out_b["z_content"], teacher_out_b["z_content"])
        loss_anchor = 0.5 * loss_anchor

    total = (
        weights["content"] * loss_content
        + weights["style"] * loss_style
        + weights["music"] * loss_music
        + weights.get("content_adv", 0.0) * loss_content_adv
        + weights.get("content_l1", 0.0) * loss_content_l1
        + weights.get("music_bias", 0.0) * loss_music_bias
        + weights.get("anchor", 0.0) * loss_anchor
    )
    return total, {
        "content": float(loss_content.detach().cpu().item()),
        "style": float(loss_style.detach().cpu().item()),
        "music": float(loss_music.detach().cpu().item()),
        "content_adv": float(loss_content_adv.detach().cpu().item()),
        "content_l1": float(loss_content_l1.detach().cpu().item()),
        "music_bias": float(loss_music_bias.detach().cpu().item()),
        "anchor": float(loss_anchor.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
        "music_pos_weight": float(pos_weight.detach().cpu().item()),
        "music_skipped": float(skip_music),
    }


def run_validation(
    model,
    loader,
    device,
    max_steps,
    weights,
    grl_lambda: float = 1.0,
    teacher_model=None,
):
    model.eval()
    stats = {
        "content": 0.0,
        "style": 0.0,
        "music": 0.0,
        "content_adv": 0.0,
        "content_l1": 0.0,
        "music_bias": 0.0,
        "anchor": 0.0,
        "total": 0.0,
        "music_pos_weight": 0.0,
        "music_skipped": 0.0,
    }
    steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_steps is not None and batch_idx > max_steps:
                break
            log_mel = batch["log_mel"].to(device, non_blocking=True)
            log_mel_aug = batch["log_mel_aug"].to(device, non_blocking=True)
            source_idx = batch["source_idx"].to(device, non_blocking=True)
            is_music = batch["is_music"].to(device, non_blocking=True)

            out_a = model(log_mel, grl_lambda=grl_lambda)
            out_b = model(log_mel_aug, grl_lambda=grl_lambda)
            teacher_out_a = None
            teacher_out_b = None
            if teacher_model is not None and float(weights.get("anchor", 0.0)) > 0.0:
                teacher_out_a = teacher_model(log_mel, grl_lambda=0.0)
                teacher_out_b = teacher_model(log_mel_aug, grl_lambda=0.0)
            _, parts = compute_losses(
                out_a,
                out_b,
                source_idx,
                is_music,
                weights,
                teacher_out_a=teacher_out_a,
                teacher_out_b=teacher_out_b,
            )
            for k in stats:
                stats[k] += parts[k]
            steps += 1

    if steps == 0:
        return {k: float("nan") for k in stats}
    return {k: stats[k] / steps for k in stats}


def make_epoch_loaders(train_ds, val_ds, cfg, phase: int, epoch: int):
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"] + phase * 100000 + epoch))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(DEVICE_PREFERENCE == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
        drop_last=True,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(DEVICE_PREFERENCE == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
        drop_last=False,
    )
    return train_loader, val_loader


def set_phase_trainable(model, phase: int, cfg: dict):
    # Default: train all branches.
    for p in model.parameters():
        p.requires_grad = True

    # Optional phase-3 gate hardening pass: train only music head.
    phase3_train_mode = str(cfg.get("phase3_train_mode", "full")).lower()
    if int(phase) == 3 and bool(cfg.get("phase3_music_head_only", False)):
        phase3_train_mode = "music_head_only"

    if int(phase) == 3 and phase3_train_mode == "music_head_only":
        freeze_modules = [
            model.backbone,
            model.shared,
            model.content_head,
            model.style_head,
            model.style_cls,
            model.content_style_adv,
        ]
        for m in freeze_modules:
            for p in m.parameters():
                p.requires_grad = False
        for p in model.music_head.parameters():
            p.requires_grad = True
    elif int(phase) == 3 and phase3_train_mode == "auc_sharpener":
        # Freeze everything first, then unfreeze only the boundary-sharpening parts.
        for p in model.parameters():
            p.requires_grad = False

        # Final conv block + pool and shared projection adapt the decision boundary
        # with minimal drift in the content branch.
        for p in model.backbone[6:].parameters():
            p.requires_grad = True
        for p in model.shared.parameters():
            p.requires_grad = True

        # Keep content/style embedding heads frozen to preserve disentanglement.
        # Only classifier heads are trainable in this mode.
        for p in model.style_cls.parameters():
            p.requires_grad = True
        for p in model.content_style_adv.parameters():
            p.requires_grad = True
        for p in model.music_head.parameters():
            p.requires_grad = True


def build_phase_cache(
    phase: int,
    source_to_idx: dict,
    cfg: dict,
    phase3_enable_hard_negatives: bool | None = None,
):
    phase_df = get_phase_manifest(
        phase,
        cfg=cfg,
        phase3_enable_hard_negatives=phase3_enable_hard_negatives,
    ).copy()
    phase_df = phase_df[phase_df["path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
    phase_df["source_idx"] = phase_df["source"].map(source_to_idx).astype(int)

    train_files, val_files = split_manifest_by_path(phase_df, val_ratio=cfg["val_ratio"], seed=cfg["seed"])

    train_chunk_df = build_chunk_index(
        train_files,
        chunk_seconds=cfg["chunk_seconds"],
        stride_seconds=cfg["stride_seconds"],
        max_files_per_source=cfg["max_files_per_source"],
        max_chunks_per_file=cfg["max_chunks_per_file"],
    )
    val_chunk_df = build_chunk_index(
        val_files,
        chunk_seconds=cfg["chunk_seconds"],
        stride_seconds=cfg["stride_seconds"],
        max_files_per_source=max(16, cfg["max_files_per_source"] // 3),
        max_chunks_per_file=max(2, cfg["max_chunks_per_file"] // 2),
    )

    train_ds = Lab1ChunkDataset(train_chunk_df, sample_rate=cfg["sample_rate"])
    val_ds = Lab1ChunkDataset(val_chunk_df, sample_rate=cfg["sample_rate"])

    return {
        "phase_df": phase_df,
        "train_chunk_df": train_chunk_df,
        "val_chunk_df": val_chunk_df,
        "train_ds": train_ds,
        "val_ds": val_ds,
    }


def set_optimizer_lrs_for_phase(optimizer, cfg: dict, phase: int, epoch: int, total_epochs: int):
    base_lr = float(cfg["lr"])
    phase3_last_n = int(cfg.get("phase3_hard_negative_last_n_epochs", 0))
    phase3_hardening_start = max(1, total_epochs - phase3_last_n + 1) if phase3_last_n > 0 else total_epochs + 1

    # Default multipliers.
    lr_mults = {
        "backbone": float(cfg.get("backbone_lr_mult", 1.0)),
        "content": 1.0,
        "music": float(cfg.get("music_lr_mult", 1.0)),
        "style": float(cfg.get("style_lr_mult", 1.0)),
        "adv": float(cfg.get("adv_lr_mult", 1.0)),
        "other": 1.0,
    }

    # Phase-3 hardening: keep backbone stable while sharpening the decision boundary.
    if int(phase) == 3 and int(epoch) >= int(phase3_hardening_start):
        lr_mults["backbone"] *= float(cfg.get("phase3_backbone_lr_scale", 0.1))

    for g in optimizer.param_groups:
        name = g.get("name", "other")
        g["lr"] = base_lr * lr_mults.get(name, 1.0)


def save_checkpoint(run_dir: Path, filename: str, model, optimizer, train_state: dict, cfg: dict, source_to_idx: dict):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_state": train_state,
        "cfg": cfg,
        "source_to_idx": source_to_idx,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    target = ckpt_dir / filename
    torch.save(payload, str(target))

    latest = run_dir / "latest.pt"
    torch.save(payload, str(latest))

    state_json = run_dir / "run_state.json"
    state_json.write_text(json.dumps(train_state, indent=2), encoding="utf-8")


def resolve_device(require_cuda: bool = True, preference: str = "cuda"):
    if preference == "cuda":
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            return "cuda"
        if require_cuda:
            raise RuntimeError(
                "CUDA is not available in the active kernel. "
                "Select kernel 'Python (lab1-venv)' and verify torch is a CUDA build."
            )
        return "cpu"
    return "cpu"


def load_latest(run_dir: Path, model, optimizer):
    latest = run_dir / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"No checkpoint found: {latest}")
    payload = torch.load(str(latest), map_location="cpu")
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. If you enabled GRL/de-style remediation, "
            "start with MODE='fresh' and a new SAVENAME."
        ) from exc
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def build_example_manifest(source_to_idx: dict, max_per_source: int = 1, seed: int = SEED) -> pd.DataFrame:
    """Fixed comparison set: one/few clips per source from phase2+phase3 manifests."""
    pools = [phase2_manifest.copy(), phase3_manifest.copy()]
    df = pd.concat(pools, ignore_index=True)
    df = df[df["path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=["source", "path", "source_idx"])

    rows = []
    for src, g in df.groupby("source"):
        if src not in source_to_idx:
            continue
        take = g.sample(min(max_per_source, len(g)), random_state=seed)
        for _, r in take.iterrows():
            rows.append({
                "source": src,
                "path": str(r["path"]),
                "source_idx": int(source_to_idx[src]),
            })
    return pd.DataFrame(rows)


def evaluate_model_examples(model, device, examples_df: pd.DataFrame, sample_rate: int, sample_seconds: float) -> pd.DataFrame:
    if len(examples_df) == 0:
        return pd.DataFrame()

    idx_to_source = {}
    if hasattr(model, "style_cls"):
        n = int(model.style_cls.out_features)
        idx_to_source = {i: f"source_{i}" for i in range(n)}

    rows = []
    model.eval()
    with torch.no_grad():
        for _, r in examples_df.iterrows():
            path = str(r["path"])
            src = str(r["source"])
            y = load_audio_chunk_48k(path=path, start_sec=0.0, duration_sec=sample_seconds, sample_rate=sample_rate)
            mel = extract_log_mel_fast(y, sr=sample_rate)
            x = torch.from_numpy(mel).unsqueeze(0).to(device, non_blocking=True)

            out = model(x)
            probs = torch.softmax(out["style_logits"], dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            topk = torch.topk(probs, k=min(3, probs.numel()))
            music_prob = float(torch.sigmoid(out["music_logit"])[0].item())

            rows.append({
                "source": src,
                "file": Path(path).name,
                "path": path,
                "music_prob": music_prob,
                "style_pred_idx": pred_idx,
                "style_pred_prob": float(probs[pred_idx].item()),
                "top1_idx": int(topk.indices[0].item()) if topk.indices.numel() > 0 else None,
                "top1_prob": float(topk.values[0].item()) if topk.values.numel() > 0 else None,
                "top2_idx": int(topk.indices[1].item()) if topk.indices.numel() > 1 else None,
                "top2_prob": float(topk.values[1].item()) if topk.values.numel() > 1 else None,
                "top3_idx": int(topk.indices[2].item()) if topk.indices.numel() > 2 else None,
                "top3_prob": float(topk.values[2].item()) if topk.values.numel() > 2 else None,
            })

    return pd.DataFrame(rows)


def render_example_comparison(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    if len(pre_df) == 0 or len(post_df) == 0:
        return pd.DataFrame()
    left = pre_df[["source", "file", "music_prob", "style_pred_idx", "style_pred_prob"]].rename(
        columns={
            "music_prob": "music_prob_before",
            "style_pred_idx": "style_pred_before",
            "style_pred_prob": "style_prob_before",
        }
    )
    right = post_df[["source", "file", "music_prob", "style_pred_idx", "style_pred_prob"]].rename(
        columns={
            "music_prob": "music_prob_after",
            "style_pred_idx": "style_pred_after",
            "style_pred_prob": "style_prob_after",
        }
    )
    out = left.merge(right, on=["source", "file"], how="inner")
    out["pred_changed"] = out["style_pred_before"] != out["style_pred_after"]
    out["music_prob_delta"] = out["music_prob_after"] - out["music_prob_before"]
    return out


def train_curriculum_resumable(cfg: dict, savename: str, mode: str, run_until_phase=None):
    assert torch is not None, "Torch is required"
    assert mode in {"fresh", "resume"}, "MODE must be 'fresh' or 'resume'"

    device = resolve_device(require_cuda=REQUIRE_CUDA, preference=DEVICE_PREFERENCE)
    phase_order = cfg["phase_order"]
    source_map_phases = cfg.get("source_map_phases", phase_order)
    source_to_idx = build_global_source_map(source_map_phases)
    if len(source_to_idx) == 0:
        raise RuntimeError("No data sources found for selected phases")

    model = ChunkEncoder(n_sources=len(source_to_idx), z_dim=cfg["z_dim"]).to(device)
    base_lr = float(cfg["lr"])
    style_lr_mult = float(cfg.get("style_lr_mult", 1.0))
    adv_lr_mult = float(cfg.get("adv_lr_mult", 1.0))
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 1.0))
    music_lr_mult = float(cfg.get("music_lr_mult", 1.0))

    backbone_params = list(model.backbone.parameters()) + list(model.shared.parameters())
    content_params = list(model.content_head.parameters())
    music_params = list(model.music_head.parameters())
    style_params = list(model.style_head.parameters()) + list(model.style_cls.parameters())
    adv_params = list(model.content_style_adv.parameters())

    known_ids = {id(p) for p in (backbone_params + content_params + music_params + style_params + adv_params)}
    other_params = [p for p in model.parameters() if id(p) not in known_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult, "name": "backbone"},
            {"params": content_params, "lr": base_lr, "name": "content"},
            {"params": music_params, "lr": base_lr * music_lr_mult, "name": "music"},
            {"params": style_params, "lr": base_lr * style_lr_mult, "name": "style"},
            {"params": adv_params, "lr": base_lr * adv_lr_mult, "name": "adv"},
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]
    )

    # Optional warm-start in fresh mode (e.g., branch-merge or phase-specific fine-tune).
    init_ckpt = cfg.get("init_checkpoint", None)
    if mode == "fresh" and init_ckpt:
        init_path = Path(str(init_ckpt))
        if not init_path.exists():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        init_payload = torch.load(str(init_path), map_location="cpu")
        init_state = init_payload.get("model", init_payload)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            f"[init] loaded model warm-start from {init_path} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    # Optional teacher anchor to prevent content drift during phase-3 sharpening.
    teacher_model = None
    if bool(cfg.get("use_teacher_anchor", False)):
        teacher_ckpt = cfg.get("teacher_anchor_checkpoint", init_ckpt)
        if teacher_ckpt is None:
            raise ValueError("use_teacher_anchor=True requires teacher_anchor_checkpoint or init_checkpoint.")
        teacher_path = Path(str(teacher_ckpt))
        if not teacher_path.exists():
            raise FileNotFoundError(f"teacher anchor checkpoint not found: {teacher_path}")
        teacher_payload = torch.load(str(teacher_path), map_location="cpu")
        teacher_state = teacher_payload.get("model", teacher_payload)
        teacher_model = ChunkEncoder(n_sources=len(source_to_idx), z_dim=cfg["z_dim"]).to(device)
        m2, u2 = teacher_model.load_state_dict(teacher_state, strict=False)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(
            f"[anchor] teacher loaded from {teacher_path} | "
            f"missing={len(m2)} unexpected={len(u2)}"
        )

    run_dir = Path.cwd() / "saves" / savename
    run_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"

    if mode == "fresh":
        # archive previous run with same savename to keep each fresh run separate
        if (run_dir / "latest.pt").exists() or (run_dir / "history.csv").exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = run_dir.parent / f"{savename}_archived_{stamp}"
            run_dir.rename(archived)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Archived old run to: {archived}")

        train_state = {
            "savename": savename,
            "mode": "fresh",
            "next_phase_idx": 0,
            "next_epoch": 1,
            "next_step": 1,
            "global_step": 0,
            "history_rows": 0,
        }
        history_df = pd.DataFrame()

        save_checkpoint(
            run_dir,
            "init.pt",
            model,
            optimizer,
            train_state=train_state,
            cfg=cfg,
            source_to_idx=source_to_idx,
        )
    else:
        payload = load_latest(run_dir, model, optimizer)
        train_state = payload["train_state"]
        if history_csv.exists():
            history_df = pd.read_csv(history_csv)
        else:
            history_df = pd.DataFrame()

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Resume state: phase_idx={train_state['next_phase_idx']} epoch={train_state['next_epoch']} step={train_state['next_step']} global_step={train_state['global_step']}")

    # fixed examples for before/after comparison
    examples_path = run_dir / "examples_manifest.csv"
    pre_examples_path = run_dir / "examples_before.csv"
    post_examples_path = run_dir / "examples_after.csv"
    compare_examples_path = run_dir / "examples_compare.csv"

    if mode == "resume" and examples_path.exists():
        examples_df = pd.read_csv(examples_path)
    else:
        examples_df = build_example_manifest(source_to_idx, max_per_source=EXAMPLE_MAX_PER_SOURCE, seed=cfg["seed"])
        examples_df.to_csv(examples_path, index=False)

    pre_examples_df = evaluate_model_examples(
        model,
        device,
        examples_df=examples_df,
        sample_rate=cfg["sample_rate"],
        sample_seconds=cfg["chunk_seconds"],
    )
    pre_examples_df.to_csv(pre_examples_path, index=False)
    print(f"Saved before-training examples: {pre_examples_path} ({len(pre_examples_df)} rows)")
    if len(pre_examples_df):
        display(pre_examples_df)

    phase_cache = {}

    start_phase_idx = int(train_state["next_phase_idx"])
    start_epoch = int(train_state["next_epoch"])
    start_step = int(train_state["next_step"])

    for p_idx in range(start_phase_idx, len(phase_order)):
        phase = int(phase_order[p_idx])

        if run_until_phase is not None and phase > int(run_until_phase):
            print(f"Stopping before phase {phase} due to RUN_UNTIL_PHASE={run_until_phase}")
            break

        if phase not in phase_cache:
            try:
                phase_cache[phase] = build_phase_cache(
                    phase,
                    source_to_idx,
                    cfg,
                    phase3_enable_hard_negatives=False,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] Phase {phase}: {e}")
                continue

        cache = phase_cache[phase]
        if len(cache["train_chunk_df"]) == 0:
            print(f"[SKIP] Phase {phase}: no training chunks")
            continue

        total_epochs = int(cfg["epochs_per_phase"].get(phase, 1))
        epoch_from = start_epoch if p_idx == start_phase_idx else 1

        print(f"\n[Phase {phase}] files={len(cache['phase_df']):,} train_chunks={len(cache['train_chunk_df']):,} val_chunks={len(cache['val_chunk_df']):,} epochs={total_epochs}")

        if phase == 3 and len(cache["phase_df"]) > 0:
            p3 = cache["phase_df"]["is_music"].value_counts().to_dict()
            print(f"phase3 balance (is_music): {p3}")

        set_phase_trainable(model, phase=phase, cfg=cfg)
        trainable_n = int(sum(p.requires_grad for p in model.parameters()))
        total_n = int(sum(1 for _ in model.parameters()))
        print(f"trainable params tensors: {trainable_n}/{total_n}")

        phase3_last_n = int(cfg.get("phase3_hard_negative_last_n_epochs", 0))
        phase3_hard_start = max(1, total_epochs - phase3_last_n + 1) if phase3_last_n > 0 else total_epochs + 1
        current_hn_state = None

        for epoch in range(epoch_from, total_epochs + 1):
            if phase == 3 and phase3_last_n > 0:
                use_hn = bool(epoch >= phase3_hard_start)
                if current_hn_state is None or use_hn != current_hn_state:
                    phase_cache[phase] = build_phase_cache(
                        phase,
                        source_to_idx,
                        cfg,
                        phase3_enable_hard_negatives=use_hn,
                    )
                    cache = phase_cache[phase]
                    current_hn_state = use_hn
                    print(
                        f"[phase3] epoch {epoch}/{total_epochs} hard_negatives={'ON' if use_hn else 'OFF'} "
                        f"train_chunks={len(cache['train_chunk_df'])}"
                    )

            set_optimizer_lrs_for_phase(
                optimizer=optimizer,
                cfg=cfg,
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
            )
            train_loader, val_loader = make_epoch_loaders(cache["train_ds"], cache["val_ds"], cfg, phase, epoch)

            max_train_steps = cfg["max_train_steps_per_epoch"]
            if max_train_steps is None:
                target_steps = len(train_loader)
            else:
                target_steps = min(int(max_train_steps), len(train_loader))

            epoch_start_step = start_step if (p_idx == start_phase_idx and epoch == epoch_from) else 1
            if epoch_start_step > target_steps:
                epoch_start_step = 1

            model.train(True)
            train_acc = {
                "content": 0.0,
                "style": 0.0,
                "music": 0.0,
                "content_adv": 0.0,
                "content_l1": 0.0,
                "music_bias": 0.0,
                "anchor": 0.0,
                "total": 0.0,
                "music_pos_weight": 0.0,
                "music_skipped": 0.0,
            }
            seen = 0
            grl_lambda = cfg.get("phase_grl_lambda", {}).get(phase, cfg.get("grl_lambda", 1.0))

            for batch_idx, batch in enumerate(train_loader, start=1):
                if batch_idx < epoch_start_step:
                    continue
                if batch_idx > target_steps:
                    break

                log_mel = batch["log_mel"].to(device, non_blocking=True)
                log_mel_aug = batch["log_mel_aug"].to(device, non_blocking=True)
                source_idx = batch["source_idx"].to(device, non_blocking=True)
                is_music = batch["is_music"].to(device, non_blocking=True)

                out_a = model(log_mel, grl_lambda=grl_lambda)
                out_b = model(log_mel_aug, grl_lambda=grl_lambda)
                phase_weights = cfg.get("phase_loss_weights", {}).get(phase, cfg["loss_weights"])
                teacher_out_a = None
                teacher_out_b = None
                if (
                    teacher_model is not None
                    and int(phase) == 3
                    and float(phase_weights.get("anchor", 0.0)) > 0.0
                ):
                    with torch.no_grad():
                        teacher_out_a = teacher_model(log_mel, grl_lambda=0.0)
                        teacher_out_b = teacher_model(log_mel_aug, grl_lambda=0.0)
                loss, parts = compute_losses(
                    out_a,
                    out_b,
                    source_idx,
                    is_music,
                    phase_weights,
                    teacher_out_a=teacher_out_a,
                    teacher_out_b=teacher_out_b,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                for k in train_acc:
                    train_acc[k] += parts[k]
                seen += 1

                train_state["global_step"] += 1
                train_state["next_phase_idx"] = p_idx
                train_state["next_epoch"] = epoch
                train_state["next_step"] = batch_idx + 1

                step_file = f"step_p{phase}_e{epoch:04d}_b{batch_idx:05d}_g{train_state['global_step']:08d}.pt"
                save_checkpoint(run_dir, step_file, model, optimizer, train_state, cfg, source_to_idx)

                if (batch_idx % cfg["print_every_steps"]) == 0:
                    print(
                        f"step {p_idx+1}/{len(phase_order)} | epoch {epoch}/{total_epochs} | "
                        f"batch {batch_idx}/{target_steps} | global_step {train_state['global_step']} | "
                        f"loss_total {parts['total']:.4f}"
                    )

            train_stats = {k: (train_acc[k] / seen if seen > 0 else float('nan')) for k in train_acc}
            phase_weights = cfg.get("phase_loss_weights", {}).get(phase, cfg["loss_weights"])
            val_stats = run_validation(
                model,
                val_loader,
                device,
                max_steps=cfg["max_val_steps_per_epoch"],
                weights=phase_weights,
                grl_lambda=grl_lambda,
                teacher_model=(teacher_model if int(phase) == 3 else None),
            )

            row = {
                "phase": phase,
                "epoch": epoch,
                "pipeline_step": p_idx + 1,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "global_step": train_state["global_step"],
            }
            history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
            history_df.to_csv(history_csv, index=False)

            # advance pointer to next epoch/phase
            if epoch < total_epochs:
                train_state["next_phase_idx"] = p_idx
                train_state["next_epoch"] = epoch + 1
                train_state["next_step"] = 1
            else:
                train_state["next_phase_idx"] = p_idx + 1
                train_state["next_epoch"] = 1
                train_state["next_step"] = 1

            train_state["history_rows"] = int(len(history_df))

            epoch_file = f"epoch_p{phase}_e{epoch:04d}_g{train_state['global_step']:08d}.pt"
            save_checkpoint(run_dir, epoch_file, model, optimizer, train_state, cfg, source_to_idx)

            print(
                f"[epoch done] step {p_idx+1}/{len(phase_order)} epoch {epoch}/{total_epochs} | "
                f"train_total={train_stats['total']:.4f} val_total={val_stats['total']:.4f}"
            )

        # reset resume offsets after first resumed phase is consumed
        start_epoch = 1
        start_step = 1

        if run_until_phase is not None and phase == int(run_until_phase):
            print(f"Reached RUN_UNTIL_PHASE={run_until_phase}. Stopping cleanly.")
            break

    post_examples_df = evaluate_model_examples(
        model,
        device,
        examples_df=examples_df,
        sample_rate=cfg["sample_rate"],
        sample_seconds=cfg["chunk_seconds"],
    )
    post_examples_df.to_csv(post_examples_path, index=False)

    compare_df = render_example_comparison(pre_examples_df, post_examples_df)
    compare_df.to_csv(compare_examples_path, index=False)

    print(f"Saved after-training examples: {post_examples_path} ({len(post_examples_df)} rows)")
    print(f"Saved before-vs-after compare: {compare_examples_path} ({len(compare_df)} rows)")
    if len(compare_df):
        display(compare_df)

    complete = int(train_state["next_phase_idx"]) >= len(phase_order)
    print("\nTraining status:", "COMPLETE" if complete else "PARTIAL")
    print(f"Next resume pointer: phase_idx={train_state['next_phase_idx']} epoch={train_state['next_epoch']} step={train_state['next_step']}")

    return model, history_df, run_dir


# ---- training config ----
CFG = {
    "seed": SEED,
    "sample_rate": 22050,
    "chunk_seconds": 5.0,
    "stride_seconds": 2.5,
    "batch_size": 4,
    "num_workers": 0,
    "val_ratio": 0.1,
    "max_files_per_source": 120,
    "max_chunks_per_file": 6,
    "phase_order": [1, 2, 3],
    "epochs_per_phase": {1: 50, 2: 50, 3: 20},
    "max_train_steps_per_epoch": 60,
    "max_val_steps_per_epoch": 20,
    "z_dim": 128,
    "lr": 1e-3,
    "loss_weights": {
        "content": 1.0,
        "style": 0.8,
        "music": 3.5,
        "content_adv": 0.55,
        "content_l1": 0.0007,
        "music_bias": 0.0005,
        "music_only_when_mixed": True,
    },
    "phase_loss_weights": {
        1: {
            "content": 1.0,
            "style": 0.55,
            "music": 0.0,
            "content_adv": 0.30,
            "content_l1": 0.0008,
            "music_bias": 0.0,
            "music_only_when_mixed": True,
        },
        2: {
            "content": 1.0,
            "style": 0.60,
            "music": 0.0,
            "content_adv": 1.00,
            "content_l1": 0.0012,
            "music_bias": 0.0,
            "music_only_when_mixed": True,
        },
        3: {
            "content": 0.8,
            "style": 0.35,
            "music": 4.0,
            "content_adv": 0.60,
            "content_l1": 0.0008,
            "music_bias": 0.0005,
            "music_only_when_mixed": True,
        },
    },
    "grl_lambda": 1.00,
    "phase_grl_lambda": {1: 0.25, 2: 1.00, 3: 0.80},
    "phase3_music_head_only": False,
    "phase3_train_mode": "full",
    "use_teacher_anchor": False,
    "teacher_anchor_checkpoint": None,
    "phase3_hard_negative_last_n_epochs": PHASE3_HARD_NEGATIVE_LAST_N_EPOCHS,
    "phase3_backbone_lr_scale": 0.05,
    "backbone_lr_mult": 1.0,
    "music_lr_mult": 2.5,
    "style_lr_mult": 2.5,
    "adv_lr_mult": 0.6,
    "print_every_steps": 1,
}

if torch is None:
    print("Torch missing. Install dependencies and re-run.")
elif RUN_TRAINING:
    model, train_history, save_dir = train_curriculum_resumable(
        cfg=CFG,
        savename=SAVENAME,
        mode=MODE,
        run_until_phase=RUN_UNTIL_PHASE,
    )
    print("save_dir:", save_dir)
    print("history_rows:", len(train_history))
    if len(train_history):
        display(train_history.tail(10))
else:
    print("RUN_TRAINING is False. Set it True after SAVENAME/MODE are configured.")




