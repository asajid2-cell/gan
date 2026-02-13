from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import EncodecModel

from .lab3_bridge import load_audio_chunk


@dataclass
class CodecConfig:
    model_id: str
    sample_rate: int
    chunk_seconds: float
    bandwidth: float
    codebook_size: int
    num_codebooks: int
    latent_channels: int
    frame_rate: float


def _to_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class FrozenEncodec:
    def __init__(
        self,
        model_id: str = "facebook/encodec_24khz",
        bandwidth: float = 6.0,
        chunk_seconds: float = 5.0,
        device: str = "auto",
    ):
        self.device = _to_device(device)
        self.model_id = str(model_id)
        self.model = EncodecModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = self.model.config
        frame_rate = getattr(self.model.quantizer, "frame_rate", None)
        if frame_rate is None:
            frame_rate = 75.0
        self.cfg = CodecConfig(
            model_id=self.model_id,
            sample_rate=int(getattr(cfg, "sampling_rate", 24000)),
            chunk_seconds=float(chunk_seconds),
            bandwidth=float(bandwidth),
            codebook_size=int(cfg.codebook_size),
            num_codebooks=int(cfg.num_quantizers),
            latent_channels=int(getattr(cfg, "hidden_size", 128)),
            frame_rate=float(frame_rate),
        )

    @torch.no_grad()
    def encode_waveforms(
        self,
        audio_wav: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        audio_wav: [B, 1, S] float32 in [-1, 1]
        Returns:
          codes: [F, B, K, L] long
          scales: [F, B, 1] float32
        """
        out = self.model.encode(
            input_values=audio_wav.to(self.device),
            bandwidth=float(self.cfg.bandwidth),
            return_dict=True,
        )
        codes = out.audio_codes.long()
        scales = out.audio_scales
        if isinstance(scales, (list, tuple)):
            scales = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s) for s in scales], dim=0)
        scales = scales.to(self.device).float()
        if scales.ndim == 2:
            scales = scales.unsqueeze(-1)
        return {
            "codes": codes,
            "scales": scales,
        }

    @torch.no_grad()
    def decode_codes(
        self,
        codes_f_b_k_l: torch.Tensor,
        scales_f_b_1: torch.Tensor,
        last_frame_pad_length: int = 0,
    ) -> torch.Tensor:
        out = self.model.decode(
            audio_codes=codes_f_b_k_l.long().to(self.device),
            audio_scales=scales_f_b_1.float().to(self.device),
            return_dict=True,
            last_frame_pad_length=int(last_frame_pad_length),
        )
        return out.audio_values

    @torch.no_grad()
    def encode_embeddings(self, audio_wav: torch.Tensor) -> torch.Tensor:
        """
        audio_wav: [B, 1, S]
        Returns quantized embeddings [B, C, T].
        """
        x = audio_wav.to(self.device).float()
        emb = self.model.encoder(x)
        codes = self.model.quantizer.encode(emb, bandwidth=float(self.cfg.bandwidth))
        q_emb = self.model.quantizer.decode(codes)
        return q_emb

    @torch.no_grad()
    def encode_codes_and_embeddings(
        self,
        audio_wav: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        audio_wav: [B, 1, S]
        Returns:
          codes: [K, B, T] long
          q_emb: [B, C, T] float32
        """
        x = audio_wav.to(self.device).float()
        emb = self.model.encoder(x)
        codes = self.model.quantizer.encode(emb, bandwidth=float(self.cfg.bandwidth))
        q_emb = self.model.quantizer.decode(codes)
        return {"codes": codes.long(), "q_emb": q_emb.float()}

    def decode_embeddings(self, q_emb_b_c_t: torch.Tensor) -> torch.Tensor:
        """
        q_emb_b_c_t: [B, C, T] float32
        Returns waveform [B, 1, S].
        """
        was_training = bool(self.model.decoder.training)
        if not was_training:
            self.model.decoder.train(True)
        out = self.model.decoder(q_emb_b_c_t.to(self.device).float())
        if not was_training:
            self.model.decoder.train(False)
        return out

    def flatten_codes(self, codes_f_b_k_l: torch.Tensor) -> torch.Tensor:
        """
        [F, B, K, L] -> [B, K, T]
        """
        f, b, k, l = codes_f_b_k_l.shape
        x = codes_f_b_k_l.permute(1, 2, 0, 3).contiguous()
        return x.view(b, k, f * l)

    def unflatten_codes(self, codes_b_k_t: torch.Tensor, n_frames: int, frame_len: int) -> torch.Tensor:
        """
        [B, K, T] -> [F, B, K, L]
        """
        b, k, t = codes_b_k_t.shape
        want = int(n_frames) * int(frame_len)
        if t != want:
            if t > want:
                codes_b_k_t = codes_b_k_t[:, :, :want]
            else:
                pad = torch.zeros((b, k, want - t), dtype=codes_b_k_t.dtype, device=codes_b_k_t.device)
                codes_b_k_t = torch.cat([codes_b_k_t, pad], dim=2)
        x = codes_b_k_t.view(b, k, int(n_frames), int(frame_len))
        return x.permute(2, 0, 1, 3).contiguous()

    def load_codec_chunk(self, path: Path, start_sec: float = 0.0) -> np.ndarray:
        """
        Loads chunk at codec sample rate.
        """
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

    def wav_to_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y.astype(np.float32)).view(1, 1, -1).to(self.device)

    def encode_chunk(self, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
          flat_codes: [K, T]
          scales: [F, 1]
          n_frames: F
          frame_len: L
        """
        wav = self.wav_to_tensor(y)
        enc = self.encode_waveforms(wav)
        codes = enc["codes"]  # [F, 1, K, L]
        scales = enc["scales"]  # [F, 1, 1]
        flat = self.flatten_codes(codes)[0]  # [K, T]
        return (
            flat.detach().cpu(),
            scales[:, 0, :].detach().cpu(),
            int(codes.shape[0]),
            int(codes.shape[-1]),
        )

    def encode_chunk_embeddings(self, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          q_emb: [C, T]
          codes: [K, T]
        """
        wav = self.wav_to_tensor(y)
        out = self.encode_codes_and_embeddings(wav)
        codes = out["codes"][:, 0, :].detach().cpu()
        q_emb = out["q_emb"][0].detach().cpu()
        return q_emb, codes

    def decode_chunk_embeddings(self, q_emb_c_t: torch.Tensor) -> np.ndarray:
        q = q_emb_c_t.view(1, q_emb_c_t.shape[0], q_emb_c_t.shape[1]).to(self.device).float()
        wav = self.decode_embeddings(q)[0, 0].detach().cpu().numpy()
        return wav.astype(np.float32)

    def expected_num_frames(self) -> int:
        return int(round(float(self.cfg.chunk_seconds) * float(self.cfg.frame_rate)))

    def target_num_samples(self) -> int:
        return int(round(float(self.cfg.chunk_seconds) * float(self.cfg.sample_rate)))

    def fix_num_frames(self, q_emb_c_t: torch.Tensor, target_frames: Optional[int] = None) -> torch.Tensor:
        want = int(target_frames if target_frames is not None else self.expected_num_frames())
        have = int(q_emb_c_t.shape[-1])
        if have == want:
            return q_emb_c_t
        if have > want:
            return q_emb_c_t[:, :want]
        pad = torch.zeros((q_emb_c_t.shape[0], want - have), dtype=q_emb_c_t.dtype, device=q_emb_c_t.device)
        return torch.cat([q_emb_c_t, pad], dim=-1)
