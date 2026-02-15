from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class _CodecLatentDataset(Dataset):
    def __init__(
        self,
        q_emb: np.ndarray,
        genre_idx: np.ndarray,
        indices: np.ndarray,
        source_idx: Optional[np.ndarray] = None,
    ):
        self.q_emb = q_emb[indices].astype(np.float32)
        self.genre_idx = genre_idx[indices].astype(np.int64)
        if source_idx is None:
            self.source_idx = np.zeros((len(self.genre_idx),), dtype=np.int64)
        else:
            self.source_idx = source_idx[indices].astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.genre_idx))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.q_emb[i]),
            torch.tensor(int(self.genre_idx[i]), dtype=torch.long),
            torch.tensor(int(self.source_idx[i]), dtype=torch.long),
        )


class _VectorDataset(Dataset):
    """Dataset for 2D feature vectors (e.g. Lab1 z_style [N, D])."""

    def __init__(
        self,
        features: np.ndarray,
        genre_idx: np.ndarray,
        indices: np.ndarray,
        source_idx: Optional[np.ndarray] = None,
    ):
        self.features = features[indices].astype(np.float32)
        self.genre_idx = genre_idx[indices].astype(np.int64)
        if source_idx is None:
            self.source_idx = np.zeros((len(self.genre_idx),), dtype=np.int64)
        else:
            self.source_idx = source_idx[indices].astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.genre_idx))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[i]),
            torch.tensor(int(self.genre_idx[i]), dtype=torch.long),
            torch.tensor(int(self.source_idx[i]), dtype=torch.long),
        )


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, float(lambd))


class CodecStyleJudge(nn.Module):
    """
    Lightweight classifier/embedding network over EnCodec quantized embeddings q_emb [B, C, T].
    Used for:
      1) gating metrics (style_conf/style_acc, diversity in embedding space)
      2) differentiable style loss for the translator (weights frozen during translator training)
    """

    def __init__(
        self,
        in_channels: int,
        n_genres: int,
        hidden: int = 256,
        emb_dim: int = 128,
        n_sources: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_genres = int(n_genres)
        self.hidden = int(hidden)
        self.emb_dim = int(emb_dim)
        self.n_sources = int(max(0, n_sources))

        self.net = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden),
            nn.SiLU(),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden),
            nn.SiLU(),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(self.hidden, self.emb_dim)
        self.head = nn.Linear(self.emb_dim, self.n_genres)
        self.source_head: Optional[nn.Module]
        if self.n_sources > 1:
            src_hidden = max(64, self.emb_dim)
            self.source_head = nn.Sequential(
                nn.Linear(self.emb_dim, src_hidden),
                nn.SiLU(),
                nn.Linear(src_hidden, self.n_sources),
            )
        else:
            self.source_head = None

    def embed(self, q_emb: torch.Tensor) -> torch.Tensor:
        h = self.net(q_emb)
        h = self.pool(h).squeeze(-1)
        e = self.proj(h)
        return F.normalize(e, dim=-1)

    def source_logits_from_embed(self, emb: torch.Tensor, grl_lambda: float = 1.0) -> Optional[torch.Tensor]:
        if self.source_head is None:
            return None
        e = _grad_reverse(emb, lambd=float(grl_lambda))
        return self.source_head(e)

    def forward(self, q_emb: torch.Tensor) -> torch.Tensor:
        e = self.embed(q_emb)
        return self.head(e)


class Lab1StyleProbe(nn.Module):
    """
    MLP probe over Lab1 z_style [B, D] that produces well-separated embeddings.
    Same role as CodecStyleJudge but operates on Lab1's 2D z_style instead of 3D q_emb.
    """

    def __init__(
        self,
        in_dim: int = 128,
        n_genres: int = 3,
        hidden: int = 256,
        emb_dim: int = 128,
        n_sources: int = 0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_genres = int(n_genres)
        self.hidden = int(hidden)
        self.emb_dim = int(emb_dim)
        self.n_sources = int(max(0, n_sources))

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
        )
        self.proj = nn.Linear(self.hidden, self.emb_dim)
        self.head = nn.Linear(self.emb_dim, self.n_genres)
        self.source_head: Optional[nn.Module]
        if self.n_sources > 1:
            src_hidden = max(64, self.emb_dim)
            self.source_head = nn.Sequential(
                nn.Linear(self.emb_dim, src_hidden),
                nn.SiLU(),
                nn.Linear(src_hidden, self.n_sources),
            )
        else:
            self.source_head = None

    def embed(self, z_style: torch.Tensor) -> torch.Tensor:
        h = self.net(z_style)
        e = self.proj(h)
        return F.normalize(e, dim=-1)

    def source_logits_from_embed(self, emb: torch.Tensor, grl_lambda: float = 1.0) -> Optional[torch.Tensor]:
        if self.source_head is None:
            return None
        e = _grad_reverse(emb, lambd=float(grl_lambda))
        return self.source_head(e)

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        e = self.embed(z_style)
        return self.head(e)


def freeze_probe(model: Lab1StyleProbe) -> Lab1StyleProbe:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class MERTStyleProbe(nn.Module):
    """MLP probe over MERT features [B, D] (typically D=768 or 1024).

    Same architecture as Lab1StyleProbe but named separately for clarity.
    """

    def __init__(
        self,
        in_dim: int = 768,
        n_genres: int = 3,
        hidden: int = 256,
        emb_dim: int = 128,
        n_sources: int = 0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_genres = int(n_genres)
        self.hidden = int(hidden)
        self.emb_dim = int(emb_dim)
        self.n_sources = int(max(0, n_sources))

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.SiLU(),
        )
        self.proj = nn.Linear(self.hidden, self.emb_dim)
        self.head = nn.Linear(self.emb_dim, self.n_genres)
        self.source_head: Optional[nn.Module]
        if self.n_sources > 1:
            src_hidden = max(64, self.emb_dim)
            self.source_head = nn.Sequential(
                nn.Linear(self.emb_dim, src_hidden),
                nn.SiLU(),
                nn.Linear(src_hidden, self.n_sources),
            )
        else:
            self.source_head = None

    def embed(self, mert_feat: torch.Tensor) -> torch.Tensor:
        h = self.net(mert_feat)
        e = self.proj(h)
        return F.normalize(e, dim=-1)

    def source_logits_from_embed(self, emb: torch.Tensor, grl_lambda: float = 1.0) -> Optional[torch.Tensor]:
        if self.source_head is None:
            return None
        e = _grad_reverse(emb, lambd=float(grl_lambda))
        return self.source_head(e)

    def forward(self, mert_feat: torch.Tensor) -> torch.Tensor:
        e = self.embed(mert_feat)
        return self.head(e)


def freeze_mert_probe(model: MERTStyleProbe) -> MERTStyleProbe:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@dataclass
class JudgeFitResult:
    best_val_acc: float
    last_val_acc: float
    train_loss: float
    best_source_val_acc: float = float("nan")
    last_source_val_acc: float = float("nan")
    source_train_loss: float = 0.0


@torch.no_grad()
def evaluate_judge_accuracy(model: CodecStyleJudge, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for q, y, _ in loader:
        q = q.to(device).float()
        y = y.to(device).long()
        logits = model(q)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct) / float(max(1, total))


@torch.no_grad()
def evaluate_source_accuracy(
    model: CodecStyleJudge,
    loader: DataLoader,
    device: torch.device,
    grl_lambda: float = 1.0,
) -> float:
    if model.source_head is None:
        return float("nan")
    model.eval()
    correct = 0
    total = 0
    for q, _, s in loader:
        q = q.to(device).float()
        s = s.to(device).long()
        emb = model.embed(q)
        src_logits = model.source_logits_from_embed(emb, grl_lambda=float(grl_lambda))
        if src_logits is None:
            continue
        pred = torch.argmax(src_logits, dim=1)
        correct += int((pred == s).sum().item())
        total += int(s.numel())
    return float(correct) / float(max(1, total))


def fit_codec_style_judge(
    arrays: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_genres: int,
    source_idx: Optional[np.ndarray],
    n_sources: int,
    device: torch.device,
    epochs: int = 6,
    lr: float = 2e-3,
    batch_size: int = 64,
    num_workers: int = 0,
    hidden: int = 256,
    emb_dim: int = 128,
    seed: int = 328,
    source_adv_weight: float = 0.0,
    source_grl_lambda: float = 1.0,
    grl_warmup_epochs: int = 0,
    patience: int = 0,
) -> Tuple[CodecStyleJudge, JudgeFitResult]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    q_emb = arrays["q_emb"]
    genre_idx = arrays["genre_idx"]
    in_channels = int(q_emb.shape[1])

    source_idx_np = None if source_idx is None else np.asarray(source_idx, dtype=np.int64)
    use_source_adv = (
        source_idx_np is not None
        and int(n_sources) > 1
        and float(source_adv_weight) > 0.0
    )

    train_ds = _CodecLatentDataset(q_emb=q_emb, genre_idx=genre_idx, indices=train_idx, source_idx=source_idx_np)
    val_ds = _CodecLatentDataset(q_emb=q_emb, genre_idx=genre_idx, indices=val_idx, source_idx=source_idx_np)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
    )

    model = CodecStyleJudge(
        in_channels=in_channels,
        n_genres=int(n_genres),
        hidden=int(hidden),
        emb_dim=int(emb_dim),
        n_sources=int(n_sources) if use_source_adv else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max(1, epochs)), eta_min=float(lr) * 0.01)

    best_val = -1.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    last_val = 0.0
    best_source_val = float("nan")
    last_source_val = float("nan")
    last_train_loss = 0.0
    last_source_train_loss = 0.0
    epochs_since_best = 0
    grl_warmup = int(max(0, grl_warmup_epochs))
    use_patience = int(max(0, patience))

    for ep in range(1, int(max(1, epochs)) + 1):
        # GRL lambda warmup: ramp from 0 to target over warmup epochs
        if use_source_adv and grl_warmup > 0 and ep <= grl_warmup:
            effective_grl = float(source_grl_lambda) * float(ep) / float(grl_warmup)
        else:
            effective_grl = float(source_grl_lambda)

        model.train()
        losses: List[float] = []
        src_losses: List[float] = []
        for q, y, s in train_loader:
            q = q.to(device).float()
            y = y.to(device).long()
            s = s.to(device).long()
            logits = model(q)
            loss_genre = F.cross_entropy(logits, y)
            loss = loss_genre
            if use_source_adv:
                emb = model.embed(q)
                src_logits = model.source_logits_from_embed(emb, grl_lambda=effective_grl)
                if src_logits is not None:
                    loss_src = F.cross_entropy(src_logits, s)
                    loss = loss + float(source_adv_weight) * loss_src
                    src_losses.append(float(loss_src.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        last_train_loss = float(np.mean(losses)) if losses else 0.0
        last_source_train_loss = float(np.mean(src_losses)) if src_losses else 0.0
        last_val = evaluate_judge_accuracy(model, val_loader, device=device)
        if use_source_adv:
            last_source_val = evaluate_source_accuracy(
                model,
                val_loader,
                device=device,
                grl_lambda=float(source_grl_lambda),
            )
        if last_val > best_val:
            best_val = float(last_val)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_source_val = float(last_source_val)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        cur_lr = scheduler.get_last_lr()[0]
        if use_source_adv:
            print(
                f"[style-judge] epoch={ep}"
                f" train_loss={last_train_loss:.4f}"
                f" src_loss={last_source_train_loss:.4f}"
                f" val_acc={last_val:.4f}"
                f" src_val_acc={last_source_val:.4f}"
                f" best={best_val:.4f}"
                f" lr={cur_lr:.2e}"
                f" grl={effective_grl:.3f}"
            )
        else:
            print(
                f"[style-judge] epoch={ep}"
                f" train_loss={last_train_loss:.4f}"
                f" val_acc={last_val:.4f}"
                f" best={best_val:.4f}"
                f" lr={cur_lr:.2e}"
            )

        if use_patience > 0 and epochs_since_best >= use_patience:
            print(f"[style-judge] early stop at epoch={ep} (patience={use_patience})")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=True)

    return model, JudgeFitResult(
        best_val_acc=float(best_val),
        last_val_acc=float(last_val),
        train_loss=float(last_train_loss),
        best_source_val_acc=float(best_source_val),
        last_source_val_acc=float(last_source_val),
        source_train_loss=float(last_source_train_loss),
    )


def fit_lab1_style_probe(
    z_style: np.ndarray,
    genre_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_genres: int,
    source_idx: Optional[np.ndarray] = None,
    n_sources: int = 0,
    device: torch.device = torch.device("cpu"),
    epochs: int = 30,
    lr: float = 2e-3,
    batch_size: int = 64,
    num_workers: int = 0,
    hidden: int = 256,
    emb_dim: int = 128,
    seed: int = 328,
    source_adv_weight: float = 0.0,
    source_grl_lambda: float = 1.0,
    grl_warmup_epochs: int = 0,
    patience: int = 8,
) -> Tuple[Lab1StyleProbe, JudgeFitResult]:
    """Train a Lab1StyleProbe on z_style [N, D] → genre classification."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    in_dim = int(z_style.shape[1])
    source_idx_np = None if source_idx is None else np.asarray(source_idx, dtype=np.int64)
    use_source_adv = (
        source_idx_np is not None
        and int(n_sources) > 1
        and float(source_adv_weight) > 0.0
    )

    train_ds = _VectorDataset(features=z_style, genre_idx=genre_idx, indices=train_idx, source_idx=source_idx_np)
    val_ds = _VectorDataset(features=z_style, genre_idx=genre_idx, indices=val_idx, source_idx=source_idx_np)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True,
                              num_workers=int(num_workers), pin_memory=bool(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False,
                            num_workers=int(num_workers), pin_memory=bool(device.type == "cuda"), drop_last=False)

    model = Lab1StyleProbe(
        in_dim=in_dim,
        n_genres=int(n_genres),
        hidden=int(hidden),
        emb_dim=int(emb_dim),
        n_sources=int(n_sources) if use_source_adv else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max(1, epochs)), eta_min=float(lr) * 0.01)

    best_val = -1.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    last_val = 0.0
    best_source_val = float("nan")
    last_source_val = float("nan")
    last_train_loss = 0.0
    last_source_train_loss = 0.0
    epochs_since_best = 0
    grl_warmup = int(max(0, grl_warmup_epochs))
    use_patience = int(max(0, patience))

    for ep in range(1, int(max(1, epochs)) + 1):
        if use_source_adv and grl_warmup > 0 and ep <= grl_warmup:
            effective_grl = float(source_grl_lambda) * float(ep) / float(grl_warmup)
        else:
            effective_grl = float(source_grl_lambda)

        model.train()
        losses: List[float] = []
        src_losses: List[float] = []
        for feat, y, s in train_loader:
            feat = feat.to(device).float()
            y = y.to(device).long()
            s = s.to(device).long()
            logits = model(feat)
            loss_genre = F.cross_entropy(logits, y)
            loss = loss_genre
            if use_source_adv:
                emb = model.embed(feat)
                src_logits = model.source_logits_from_embed(emb, grl_lambda=effective_grl)
                if src_logits is not None:
                    loss_src = F.cross_entropy(src_logits, s)
                    loss = loss + float(source_adv_weight) * loss_src
                    src_losses.append(float(loss_src.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        last_train_loss = float(np.mean(losses)) if losses else 0.0
        last_source_train_loss = float(np.mean(src_losses)) if src_losses else 0.0

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feat, y, s in val_loader:
                feat = feat.to(device).float()
                y = y.to(device).long()
                logits = model(feat)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        last_val = float(correct) / float(max(1, total))

        if use_source_adv and model.source_head is not None:
            src_correct = src_total = 0
            with torch.no_grad():
                for feat, _, s in val_loader:
                    feat = feat.to(device).float()
                    s = s.to(device).long()
                    emb = model.embed(feat)
                    src_logits = model.source_logits_from_embed(emb, grl_lambda=float(source_grl_lambda))
                    if src_logits is not None:
                        pred = torch.argmax(src_logits, dim=1)
                        src_correct += int((pred == s).sum().item())
                        src_total += int(s.numel())
            last_source_val = float(src_correct) / float(max(1, src_total))

        if last_val > best_val:
            best_val = float(last_val)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_source_val = float(last_source_val)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"[lab1-probe] epoch={ep}"
            f" train_loss={last_train_loss:.4f}"
            f" val_acc={last_val:.4f}"
            f" best={best_val:.4f}"
            f" lr={cur_lr:.2e}"
            + (f" grl={effective_grl:.3f} src_val={last_source_val:.4f}" if use_source_adv else "")
        )

        if use_patience > 0 and epochs_since_best >= use_patience:
            print(f"[lab1-probe] early stop at epoch={ep} (patience={use_patience})")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=True)

    return model, JudgeFitResult(
        best_val_acc=float(best_val),
        last_val_acc=float(last_val),
        train_loss=float(last_train_loss),
        best_source_val_acc=float(best_source_val),
        last_source_val_acc=float(last_source_val),
        source_train_loss=float(last_source_train_loss),
    )


def fit_mert_style_probe(
    mert_feat: np.ndarray,
    genre_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_genres: int,
    source_idx: Optional[np.ndarray] = None,
    n_sources: int = 0,
    device: torch.device = torch.device("cpu"),
    epochs: int = 30,
    lr: float = 2e-3,
    batch_size: int = 64,
    num_workers: int = 0,
    hidden: int = 256,
    emb_dim: int = 128,
    seed: int = 328,
    source_adv_weight: float = 0.0,
    source_grl_lambda: float = 1.0,
    grl_warmup_epochs: int = 0,
    patience: int = 8,
) -> Tuple[MERTStyleProbe, JudgeFitResult]:
    """Train a MERTStyleProbe on mert_feat [N, D] -> genre classification."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    in_dim = int(mert_feat.shape[1])
    source_idx_np = None if source_idx is None else np.asarray(source_idx, dtype=np.int64)
    use_source_adv = (
        source_idx_np is not None
        and int(n_sources) > 1
        and float(source_adv_weight) > 0.0
    )

    train_ds = _VectorDataset(features=mert_feat, genre_idx=genre_idx, indices=train_idx, source_idx=source_idx_np)
    val_ds = _VectorDataset(features=mert_feat, genre_idx=genre_idx, indices=val_idx, source_idx=source_idx_np)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True,
                              num_workers=int(num_workers), pin_memory=bool(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False,
                            num_workers=int(num_workers), pin_memory=bool(device.type == "cuda"), drop_last=False)

    model = MERTStyleProbe(
        in_dim=in_dim,
        n_genres=int(n_genres),
        hidden=int(hidden),
        emb_dim=int(emb_dim),
        n_sources=int(n_sources) if use_source_adv else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max(1, epochs)), eta_min=float(lr) * 0.01)

    best_val = -1.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    last_val = 0.0
    best_source_val = float("nan")
    last_source_val = float("nan")
    last_train_loss = 0.0
    last_source_train_loss = 0.0
    epochs_since_best = 0
    grl_warmup = int(max(0, grl_warmup_epochs))
    use_patience = int(max(0, patience))

    for ep in range(1, int(max(1, epochs)) + 1):
        if use_source_adv and grl_warmup > 0 and ep <= grl_warmup:
            effective_grl = float(source_grl_lambda) * float(ep) / float(grl_warmup)
        else:
            effective_grl = float(source_grl_lambda)

        model.train()
        losses: List[float] = []
        src_losses: List[float] = []
        for feat, y, s in train_loader:
            feat = feat.to(device).float()
            y = y.to(device).long()
            s = s.to(device).long()
            logits = model(feat)
            loss_genre = F.cross_entropy(logits, y)
            loss = loss_genre
            if use_source_adv:
                emb = model.embed(feat)
                src_logits = model.source_logits_from_embed(emb, grl_lambda=effective_grl)
                if src_logits is not None:
                    loss_src = F.cross_entropy(src_logits, s)
                    loss = loss + float(source_adv_weight) * loss_src
                    src_losses.append(float(loss_src.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        last_train_loss = float(np.mean(losses)) if losses else 0.0
        last_source_train_loss = float(np.mean(src_losses)) if src_losses else 0.0

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for feat, y, s in val_loader:
                feat = feat.to(device).float()
                y = y.to(device).long()
                logits = model(feat)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        last_val = float(correct) / float(max(1, total))

        if use_source_adv and model.source_head is not None:
            src_correct = src_total = 0
            with torch.no_grad():
                for feat, _, s in val_loader:
                    feat = feat.to(device).float()
                    s = s.to(device).long()
                    emb = model.embed(feat)
                    src_logits = model.source_logits_from_embed(emb, grl_lambda=float(source_grl_lambda))
                    if src_logits is not None:
                        pred = torch.argmax(src_logits, dim=1)
                        src_correct += int((pred == s).sum().item())
                        src_total += int(s.numel())
            last_source_val = float(src_correct) / float(max(1, src_total))

        if last_val > best_val:
            best_val = float(last_val)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_source_val = float(last_source_val)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"[mert-probe] epoch={ep}"
            f" train_loss={last_train_loss:.4f}"
            f" val_acc={last_val:.4f}"
            f" best={best_val:.4f}"
            f" lr={cur_lr:.2e}"
            + (f" grl={effective_grl:.3f} src_val={last_source_val:.4f}" if use_source_adv else "")
        )

        if use_patience > 0 and epochs_since_best >= use_patience:
            print(f"[mert-probe] early stop at epoch={ep} (patience={use_patience})")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=True)

    return model, JudgeFitResult(
        best_val_acc=float(best_val),
        last_val_acc=float(last_val),
        train_loss=float(last_train_loss),
        best_source_val_acc=float(best_source_val),
        last_source_val_acc=float(last_source_val),
        source_train_loss=float(last_source_train_loss),
    )


@dataclass
class SourceRemovalResult:
    projection: np.ndarray  # [C, C] null-space projection matrix
    source_acc_before: float
    source_acc_after: float
    genre_acc_before: float
    genre_acc_after: float
    n_removed_dims: int


def fit_source_removal_projection(
    q_emb: np.ndarray,
    source_idx: np.ndarray,
    genre_idx: np.ndarray,
    n_sources: int,
    seed: int = 328,
    test_size: float = 0.20,
    max_remove_frac: float = 0.25,
    max_iterations: int = 20,
) -> SourceRemovalResult:
    """
    Iterative Null-space Projection (INLP) for source removal.

    Repeatedly fits a linear source predictor on q_emb.mean(axis=2), removes
    its discriminative directions, and refits until source_acc drops near chance
    or genre_acc degrades too much. Each round removes up to n_sources-1 new
    directions, accumulating a combined projection matrix.

    max_remove_frac: upper bound on total fraction of dims removed (hard stop).
    max_iterations: max INLP rounds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    C = int(q_emb.shape[1])
    max_remove_total = max(1, int(float(max_remove_frac) * C))
    q_mean = q_emb.mean(axis=2).astype(np.float64)  # [N, C]

    # Split for evaluation
    x_tr, x_te, ys_tr, ys_te, yg_tr, yg_te = train_test_split(
        q_mean, source_idx, genre_idx,
        test_size=float(test_size), random_state=int(seed), stratify=source_idx,
    )

    # Measure baseline
    clf_src_before = LogisticRegression(max_iter=2000, random_state=int(seed))
    clf_src_before.fit(x_tr, ys_tr)
    source_acc_before = float(clf_src_before.score(x_te, ys_te))

    clf_gen_before = LogisticRegression(max_iter=2000, random_state=int(seed))
    clf_gen_before.fit(x_tr, yg_tr)
    genre_acc_before = float(clf_gen_before.score(x_te, yg_te))

    source_chance = float(np.max(np.bincount(ys_te.astype(np.int64))) / len(ys_te))

    # Accumulate projection: start with identity
    P = np.eye(C, dtype=np.float64)
    removed_total = 0
    x_tr_cur = x_tr.copy()
    x_te_cur = x_te.copy()

    for it in range(1, int(max_iterations) + 1):
        if removed_total >= max_remove_total:
            break

        clf = LogisticRegression(max_iter=2000, random_state=int(seed) + it)
        clf.fit(x_tr_cur, ys_tr)
        cur_src_acc = float(clf.score(x_te_cur, ys_te))

        # Stop if source acc is near chance
        if cur_src_acc <= source_chance + 0.03:
            print(f"[source-removal] iter={it} source_acc={cur_src_acc:.4f} near chance={source_chance:.4f}, stopping")
            break

        W = clf.coef_.astype(np.float64)  # [n_classes, C_effective]
        _, _, Vt = np.linalg.svd(W, full_matrices=False)

        # How many to remove this round (capped by budget)
        n_this = min(int(Vt.shape[0]), max_remove_total - removed_total)
        if n_this <= 0:
            break
        V_src = Vt[:n_this, :].T  # [C_effective, n_this]
        P_round = np.eye(x_tr_cur.shape[1], dtype=np.float64) - V_src @ V_src.T

        # Apply this round's projection
        x_tr_cur = x_tr_cur @ P_round.T
        x_te_cur = x_te_cur @ P_round.T

        # Compose into cumulative projection
        P = P_round @ P
        removed_total += n_this

        # Check genre acc to guard against over-removal
        clf_g = LogisticRegression(max_iter=2000, random_state=int(seed))
        clf_g.fit(x_tr_cur, yg_tr)
        cur_gen_acc = float(clf_g.score(x_te_cur, yg_te))

        clf_s = LogisticRegression(max_iter=2000, random_state=int(seed))
        clf_s.fit(x_tr_cur, ys_tr)
        after_src_acc = float(clf_s.score(x_te_cur, ys_te))

        print(
            f"[source-removal] iter={it} removed={n_this} total_removed={removed_total}/{C}"
            f"  source_acc={after_src_acc:.4f} genre_acc={cur_gen_acc:.4f}"
        )

        # Stop if genre dropped too much relative to source improvement
        genre_drop = genre_acc_before - cur_gen_acc
        if genre_drop > 0.15:
            print(f"[source-removal] genre_acc dropped {genre_drop:.4f} > 0.15, stopping")
            break

    # Final measurement
    source_acc_after = float(LogisticRegression(max_iter=2000, random_state=int(seed)).fit(x_tr_cur, ys_tr).score(x_te_cur, ys_te))
    genre_acc_after = float(LogisticRegression(max_iter=2000, random_state=int(seed)).fit(x_tr_cur, yg_tr).score(x_te_cur, yg_te))

    print(
        f"[source-removal] FINAL removed {removed_total}/{C} dims"
        f"  source_acc: {source_acc_before:.4f} -> {source_acc_after:.4f}"
        f"  genre_acc: {genre_acc_before:.4f} -> {genre_acc_after:.4f}"
    )

    return SourceRemovalResult(
        projection=P.astype(np.float32),
        source_acc_before=source_acc_before,
        source_acc_after=source_acc_after,
        genre_acc_before=genre_acc_before,
        genre_acc_after=genre_acc_after,
        n_removed_dims=removed_total,
    )


def apply_source_removal_to_q_emb(
    q_emb: np.ndarray,
    projection: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """Apply [C,C] projection to q_emb [N, C, T] along channel dim. Chunked to save memory."""
    P = projection.astype(np.float32)
    N = int(q_emb.shape[0])
    out = np.empty_like(q_emb)
    for start in range(0, N, int(chunk_size)):
        end = min(start + int(chunk_size), N)
        # q_emb[n, :, t] -> P @ q_emb[n, :, t]
        out[start:end] = np.einsum("ij,njt->nit", P, q_emb[start:end].astype(np.float32))
    return out


def freeze_judge(model: CodecStyleJudge) -> CodecStyleJudge:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
