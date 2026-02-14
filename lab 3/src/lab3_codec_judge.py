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

    best_val = -1.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    last_val = 0.0
    best_source_val = float("nan")
    last_source_val = float("nan")
    last_train_loss = 0.0
    last_source_train_loss = 0.0

    for ep in range(1, int(max(1, epochs)) + 1):
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
                src_logits = model.source_logits_from_embed(emb, grl_lambda=float(source_grl_lambda))
                if src_logits is not None:
                    loss_src = F.cross_entropy(src_logits, s)
                    loss = loss + float(source_adv_weight) * loss_src
                    src_losses.append(float(loss_src.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

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

        if use_source_adv:
            print(
                f"[style-judge] epoch={ep}"
                f" train_loss={last_train_loss:.4f}"
                f" src_loss={last_source_train_loss:.4f}"
                f" val_acc={last_val:.4f}"
                f" src_val_acc={last_source_val:.4f}"
                f" best={best_val:.4f}"
            )
        else:
            print(f"[style-judge] epoch={ep} train_loss={last_train_loss:.4f} val_acc={last_val:.4f} best={best_val:.4f}")

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


def freeze_judge(model: CodecStyleJudge) -> CodecStyleJudge:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
