from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor

from .lab3_bridge import load_audio_chunk


@dataclass
class MERTConfig:
    model_id: str
    sample_rate: int
    hidden_size: int
    num_hidden_layers: int
    chunk_seconds: float


def _to_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class FrozenMERT:
    """Frozen MERT wrapper for extracting music understanding features.

    Produces mean-pooled hidden-state vectors [B, D] from audio waveforms,
    suitable for genre-level style conditioning.
    """

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-95M",
        chunk_seconds: float = 5.0,
        device: str = "auto",
        layer: int = -1,
    ):
        self.device = _to_device(device)
        self.model_id = str(model_id)
        self.layer = int(layer)

        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = self.model.config
        self.cfg = MERTConfig(
            model_id=self.model_id,
            sample_rate=int(self.processor.sampling_rate),
            hidden_size=int(cfg.hidden_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            chunk_seconds=float(chunk_seconds),
        )

    @torch.no_grad()
    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract mean-pooled features from a mono waveform (1-D float32).

        Args:
            waveform: [S] float32 at self.cfg.sample_rate

        Returns:
            features: [D] float32 mean-pooled hidden states
        """
        inputs = self.processor(
            waveform,
            sampling_rate=int(self.cfg.sample_rate),
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self.device)
        else:
            attention_mask = None

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        layer_idx = self.layer if self.layer >= 0 else len(hidden_states) + self.layer
        layer_idx = max(0, min(layer_idx, len(hidden_states) - 1))
        h = hidden_states[layer_idx]  # [1, T, D]

        feat = h.mean(dim=1).squeeze(0)  # [D]
        return feat.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_features_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Extract features for a batch of waveforms.

        Args:
            waveforms: list of [S] float32 arrays at self.cfg.sample_rate

        Returns:
            features: [B, D] float32
        """
        inputs = self.processor(
            waveforms,
            sampling_rate=int(self.cfg.sample_rate),
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        layer_idx = self.layer if self.layer >= 0 else len(hidden_states) + self.layer
        layer_idx = max(0, min(layer_idx, len(hidden_states) - 1))
        h = hidden_states[layer_idx]  # [B, T, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            # mask may be shorter than h if processor didn't produce full-length mask
            if mask.shape[1] < h.shape[1]:
                pad = torch.ones((mask.shape[0], h.shape[1] - mask.shape[1], 1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, pad], dim=1)
            elif mask.shape[1] > h.shape[1]:
                mask = mask[:, : h.shape[1], :]
            feat = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            feat = h.mean(dim=1)

        return feat.detach().cpu().numpy().astype(np.float32)

    def load_mert_chunk(self, path: Path, start_sec: float = 0.0) -> np.ndarray:
        """Load audio chunk at MERT sample rate."""
        return load_audio_chunk(
            path=Path(path),
            sample_rate=int(self.cfg.sample_rate),
            seconds=float(self.cfg.chunk_seconds),
            start_sec=float(start_sec),
        ).astype(np.float32)

    @staticmethod
    def resample_audio(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if int(sr_from) == int(sr_to):
            return y.astype(np.float32)
        z = librosa.resample(y.astype(np.float32), orig_sr=int(sr_from), target_sr=int(sr_to), res_type="soxr_hq")
        return z.astype(np.float32)
