from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PianoRollModelConfig:
    in_channels: int = 17
    hidden_channels: int = 96
    n_keys: int = 88
    n_blocks: int = 6
    dropout: float = 0.05
    architecture: str = "conv1d"
    key_embed_dim: int = 32


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=int(dilation), dilation=int(dilation))
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=max(1, channels // 16), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=max(1, channels // 16), num_channels=channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return x + h


class PianoRollGenerator(nn.Module):
    """Small source-conditioning to piano-roll generator for smoke training."""

    def __init__(self, cfg: PianoRollModelConfig = PianoRollModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Conv1d(int(cfg.in_channels), int(cfg.hidden_channels), kernel_size=5, padding=2)
        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(int(cfg.hidden_channels), dilations[i % len(dilations)], float(cfg.dropout))
                for i in range(int(cfg.n_blocks))
            ]
        )
        self.out_norm = nn.GroupNorm(
            num_groups=max(1, int(cfg.hidden_channels) // 16),
            num_channels=int(cfg.hidden_channels),
        )
        self.architecture = str(cfg.architecture or "conv1d").lower()
        if self.architecture in {
            "key_conditioned",
            "chroma_key_conditioned",
            "harmony_conditioned",
            "musical_plan_conditioned",
        }:
            key_dim = max(4, int(cfg.key_embed_dim))
            self.time_key_proj = nn.Conv1d(int(cfg.hidden_channels), key_dim, kernel_size=1)
            self.key_embedding = nn.Embedding(int(cfg.n_keys), key_dim)
            if self.architecture in {"chroma_key_conditioned", "harmony_conditioned", "musical_plan_conditioned"}:
                self.source_chroma_key_proj = nn.Linear(1, key_dim)
            if self.architecture in {"harmony_conditioned", "musical_plan_conditioned"}:
                self.harmony_head = nn.Conv1d(int(cfg.hidden_channels), 12, kernel_size=1)
                self.harmony_key_proj = nn.Linear(1, key_dim)
            if self.architecture == "musical_plan_conditioned":
                self.chord_key_proj = nn.Linear(1, key_dim)
                self.bass_key_proj = nn.Linear(1, key_dim)
                self.voicing_key_proj = nn.Linear(5, key_dim)
                self.event_key_proj = nn.Linear(4, key_dim)
                self.pc_onset_key_proj = nn.Linear(1, key_dim)
                self.role_key_proj = nn.Linear(6, key_dim)
                nn.init.zeros_(self.role_key_proj.weight)
                nn.init.zeros_(self.role_key_proj.bias)
                self.melody_key_proj = nn.Linear(5, key_dim)
                nn.init.zeros_(self.melody_key_proj.weight)
                nn.init.zeros_(self.melody_key_proj.bias)
                self.texture_role_key_proj = nn.Linear(5, key_dim)
                nn.init.zeros_(self.texture_role_key_proj.weight)
                nn.init.zeros_(self.texture_role_key_proj.bias)
                self.section_role_key_proj = nn.Linear(5, key_dim)
                nn.init.zeros_(self.section_role_key_proj.weight)
                nn.init.zeros_(self.section_role_key_proj.bias)
                self.arranger_state_key_proj = nn.Linear(9, key_dim)
                nn.init.zeros_(self.arranger_state_key_proj.weight)
                nn.init.zeros_(self.arranger_state_key_proj.bias)
                self.bass_continuity_key_proj = nn.Linear(5, key_dim)
                nn.init.zeros_(self.bass_continuity_key_proj.weight)
                nn.init.zeros_(self.bass_continuity_key_proj.bias)
                self.body_melody_state_key_proj = nn.Linear(7, key_dim)
                nn.init.zeros_(self.body_melody_state_key_proj.weight)
                nn.init.zeros_(self.body_melody_state_key_proj.bias)
                self.section_diversity_key_proj = nn.Linear(5, key_dim)
                nn.init.zeros_(self.section_diversity_key_proj.weight)
                nn.init.zeros_(self.section_diversity_key_proj.bias)
            self.onset_key_head = nn.Linear(key_dim, 1)
            self.frame_key_head = nn.Linear(key_dim, 1)
            self.velocity_key_head = nn.Linear(key_dim, 1)
        elif self.architecture == "conv1d":
            self.onset_head = nn.Conv1d(int(cfg.hidden_channels), int(cfg.n_keys), kernel_size=1)
            self.frame_head = nn.Conv1d(int(cfg.hidden_channels), int(cfg.n_keys), kernel_size=1)
            self.velocity_head = nn.Conv1d(int(cfg.hidden_channels), int(cfg.n_keys), kernel_size=1)
        else:
            raise ValueError(f"Unknown PianoRollGenerator architecture: {cfg.architecture}")
        self.pedal_head = nn.Conv1d(int(cfg.hidden_channels), 1, kernel_size=1)
        self.density_head = nn.Conv1d(int(cfg.hidden_channels), 2, kernel_size=1)
        self.register_head = nn.Conv1d(int(cfg.hidden_channels), 3, kernel_size=1)
        self.chord_head = nn.Conv1d(int(cfg.hidden_channels), 13, kernel_size=1)
        self.bass_head = nn.Conv1d(int(cfg.hidden_channels), 13, kernel_size=1)
        self.voicing_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.event_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.pc_onset_head = nn.Conv1d(int(cfg.hidden_channels), 12, kernel_size=1)
        self.role_head = nn.Conv1d(int(cfg.hidden_channels), 5, kernel_size=1)
        self.melody_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.texture_role_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.section_role_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.arranger_state_head = nn.Conv1d(int(cfg.hidden_channels), 8, kernel_size=1)
        self.bass_continuity_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)
        self.body_melody_state_head = nn.Conv1d(int(cfg.hidden_channels), 6, kernel_size=1)
        self.section_diversity_head = nn.Conv1d(int(cfg.hidden_channels), 4, kernel_size=1)

    def forward(self, source_condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.silu(self.in_proj(source_condition))
        for block in self.blocks:
            h = block(h)
        h = F.silu(self.out_norm(h))
        if self.architecture in {
            "key_conditioned",
            "chroma_key_conditioned",
            "harmony_conditioned",
            "musical_plan_conditioned",
        }:
            time_feat = self.time_key_proj(h)
            key_ids = torch.arange(int(self.cfg.n_keys), device=h.device, dtype=torch.long)
            key_feat = self.key_embedding(key_ids)
            combined_bt = (time_feat[:, None, :, :] + key_feat[None, :, :, None]).permute(0, 1, 3, 2)
            harmony_plan = None
            if self.architecture in {"chroma_key_conditioned", "harmony_conditioned", "musical_plan_conditioned"}:
                pitch_class = torch.remainder(key_ids + 21, 12)
                source_chroma = torch.clamp(source_condition[:, pitch_class, :], 0.0, 1.0)
                combined_bt = combined_bt + self.source_chroma_key_proj(source_chroma.unsqueeze(-1))
            if self.architecture in {"harmony_conditioned", "musical_plan_conditioned"}:
                harmony_plan = torch.sigmoid(self.harmony_head(h))
                key_harmony = torch.clamp(harmony_plan[:, pitch_class, :], 0.0, 1.0)
                combined_bt = combined_bt + self.harmony_key_proj(key_harmony.unsqueeze(-1))
            chord_logits = self.chord_head(h)
            bass_logits = self.bass_head(h)
            voicing_plan = torch.sigmoid(self.voicing_head(h))
            event_plan = torch.sigmoid(self.event_head(h))
            pc_onset_logits = self.pc_onset_head(h)
            pc_onset_plan = torch.sigmoid(pc_onset_logits)
            role_plan = torch.sigmoid(self.role_head(h))
            melody_plan = torch.sigmoid(self.melody_head(h))
            texture_role_plan = torch.sigmoid(self.texture_role_head(h))
            section_role_plan = torch.sigmoid(self.section_role_head(h))
            arranger_state_plan = torch.sigmoid(self.arranger_state_head(h))
            bass_continuity_plan = torch.sigmoid(self.bass_continuity_head(h))
            body_melody_state_plan = torch.sigmoid(self.body_melody_state_head(h))
            section_diversity_plan = torch.sigmoid(self.section_diversity_head(h))
            if self.architecture == "musical_plan_conditioned":
                chord_plan = torch.sigmoid(chord_logits)
                bass_plan = torch.softmax(bass_logits, dim=1)
                key_chord = torch.clamp(chord_plan[:, pitch_class, :], 0.0, 1.0)
                key_bass = torch.clamp(bass_plan[:, pitch_class, :], 0.0, 1.0)
                key_pc_onset = torch.clamp(pc_onset_plan[:, pitch_class, :], 0.0, 1.0)
                key_norm = (
                    torch.arange(int(self.cfg.n_keys), device=h.device, dtype=voicing_plan.dtype)
                    / max(1.0, float(int(self.cfg.n_keys) - 1))
                )
                key_norm_bt = key_norm[None, :, None, None].expand(
                    voicing_plan.shape[0], -1, voicing_plan.shape[2], 1
                )
                voicing_bt = voicing_plan.permute(0, 2, 1)[:, None, :, :].expand(-1, int(self.cfg.n_keys), -1, -1)
                combined_bt = combined_bt + self.chord_key_proj(key_chord.unsqueeze(-1))
                combined_bt = combined_bt + self.bass_key_proj(key_bass.unsqueeze(-1))
                combined_bt = combined_bt + self.voicing_key_proj(torch.cat([voicing_bt, key_norm_bt], dim=-1))
                event_bt = event_plan.permute(0, 2, 1)[:, None, :, :].expand(-1, int(self.cfg.n_keys), -1, -1)
                combined_bt = combined_bt + self.event_key_proj(event_bt)
                combined_bt = combined_bt + self.pc_onset_key_proj(key_pc_onset.unsqueeze(-1))
                role_bt = role_plan.permute(0, 2, 1)[:, None, :, :].expand(-1, int(self.cfg.n_keys), -1, -1)
                combined_bt = combined_bt + self.role_key_proj(torch.cat([role_bt, key_norm_bt], dim=-1))
                melody_bt = melody_plan.permute(0, 2, 1)[:, None, :, :].expand(-1, int(self.cfg.n_keys), -1, -1)
                combined_bt = combined_bt + self.melody_key_proj(torch.cat([melody_bt, key_norm_bt], dim=-1))
                texture_role_bt = texture_role_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.texture_role_key_proj(torch.cat([texture_role_bt, key_norm_bt], dim=-1))
                section_role_bt = section_role_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.section_role_key_proj(torch.cat([section_role_bt, key_norm_bt], dim=-1))
                arranger_state_bt = arranger_state_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.arranger_state_key_proj(
                    torch.cat([arranger_state_bt, key_norm_bt], dim=-1)
                )
                bass_continuity_bt = bass_continuity_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.bass_continuity_key_proj(
                    torch.cat([bass_continuity_bt, key_norm_bt], dim=-1)
                )
                body_melody_state_bt = body_melody_state_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.body_melody_state_key_proj(
                    torch.cat([body_melody_state_bt, key_norm_bt], dim=-1)
                )
                section_diversity_bt = section_diversity_plan.permute(0, 2, 1)[:, None, :, :].expand(
                    -1,
                    int(self.cfg.n_keys),
                    -1,
                    -1,
                )
                combined_bt = combined_bt + self.section_diversity_key_proj(
                    torch.cat([section_diversity_bt, key_norm_bt], dim=-1)
                )
            combined_bt = torch.tanh(combined_bt)
            out = {
                "onset_logits": self.onset_key_head(combined_bt).squeeze(-1),
                "frame_logits": self.frame_key_head(combined_bt).squeeze(-1),
                "velocity": torch.sigmoid(self.velocity_key_head(combined_bt).squeeze(-1)),
                "pedal": torch.sigmoid(self.pedal_head(h)).squeeze(1),
                "density": torch.sigmoid(self.density_head(h)),
                "register": torch.sigmoid(self.register_head(h)),
                "chord_logits": chord_logits,
                "bass_logits": bass_logits,
                "voicing": voicing_plan,
                "event": event_plan,
                "pc_onset_logits": pc_onset_logits,
                "pc_onset": pc_onset_plan,
                "role": role_plan,
                "melody": melody_plan,
                "texture_role": texture_role_plan,
                "section_role": section_role_plan,
                "arranger_state": arranger_state_plan,
                "bass_continuity": bass_continuity_plan,
                "body_melody_state": body_melody_state_plan,
                "section_diversity": section_diversity_plan,
            }
            if harmony_plan is not None:
                out["harmony"] = harmony_plan
            return out
        pc_onset_logits = self.pc_onset_head(h)
        return {
            "onset_logits": self.onset_head(h),
            "frame_logits": self.frame_head(h),
            "velocity": torch.sigmoid(self.velocity_head(h)),
            "pedal": torch.sigmoid(self.pedal_head(h)).squeeze(1),
            "density": torch.sigmoid(self.density_head(h)),
            "register": torch.sigmoid(self.register_head(h)),
            "chord_logits": self.chord_head(h),
            "bass_logits": self.bass_head(h),
            "voicing": torch.sigmoid(self.voicing_head(h)),
            "event": torch.sigmoid(self.event_head(h)),
            "pc_onset_logits": pc_onset_logits,
            "pc_onset": torch.sigmoid(pc_onset_logits),
            "role": torch.sigmoid(self.role_head(h)),
            "melody": torch.sigmoid(self.melody_head(h)),
            "texture_role": torch.sigmoid(self.texture_role_head(h)),
            "section_role": torch.sigmoid(self.section_role_head(h)),
            "arranger_state": torch.sigmoid(self.arranger_state_head(h)),
            "bass_continuity": torch.sigmoid(self.bass_continuity_head(h)),
            "body_melody_state": torch.sigmoid(self.body_melody_state_head(h)),
            "section_diversity": torch.sigmoid(self.section_diversity_head(h)),
        }


def piano_roll_loss(
    pred: Dict[str, torch.Tensor],
    target_onset: torch.Tensor,
    target_frame: torch.Tensor,
    target_velocity: torch.Tensor,
    target_pedal: torch.Tensor,
    *,
    target_density: torch.Tensor | None = None,
    target_register: torch.Tensor | None = None,
    target_chord: torch.Tensor | None = None,
    target_bass: torch.Tensor | None = None,
    target_voicing: torch.Tensor | None = None,
    target_event: torch.Tensor | None = None,
    target_pc_onset: torch.Tensor | None = None,
    target_role: torch.Tensor | None = None,
    target_melody: torch.Tensor | None = None,
    target_texture_role: torch.Tensor | None = None,
    target_section_role: torch.Tensor | None = None,
    target_arranger_state: torch.Tensor | None = None,
    target_bass_continuity: torch.Tensor | None = None,
    target_body_melody_state: torch.Tensor | None = None,
    target_section_diversity: torch.Tensor | None = None,
    source_onset: torch.Tensor | None = None,
    source_chroma: torch.Tensor | None = None,
    density_weight: float = 0.0,
    chroma_weight: float = 0.0,
    pitch_usage_weight: float = 0.0,
    hierarchy_weight: float = 0.0,
    musical_plan_weight: float = 0.0,
    event_plan_weight: float = 0.0,
    pc_onset_plan_weight: float = 0.0,
    pc_onset_f1_weight: float = 0.0,
    pc_onset_alignment_weight: float = 0.0,
    role_plan_weight: float = 0.0,
    texture_balance_weight: float = 0.0,
    melody_plan_weight: float = 0.0,
    melody_balance_weight: float = 0.0,
    texture_role_plan_weight: float = 0.0,
    texture_role_balance_weight: float = 0.0,
    section_role_plan_weight: float = 0.0,
    section_role_balance_weight: float = 0.0,
    arranger_state_plan_weight: float = 0.0,
    bass_continuity_plan_weight: float = 0.0,
    body_melody_state_plan_weight: float = 0.0,
    body_melody_state_balance_weight: float = 0.0,
    section_diversity_plan_weight: float = 0.0,
    section_diversity_balance_weight: float = 0.0,
    anti_collapse_weight: float = 0.0,
    source_onset_weight: float = 0.0,
    source_chroma_weight: float = 0.0,
    harmonic_plan_weight: float = 0.0,
    piano_min_midi: int = 21,
) -> Dict[str, torch.Tensor]:
    onset_pos = torch.clamp((target_onset.numel() - target_onset.sum()) / (target_onset.sum() + 1.0), 1.0, 80.0)
    frame_pos = torch.clamp((target_frame.numel() - target_frame.sum()) / (target_frame.sum() + 1.0), 1.0, 20.0)
    onset_loss = F.binary_cross_entropy_with_logits(pred["onset_logits"], target_onset, pos_weight=onset_pos)
    frame_loss = F.binary_cross_entropy_with_logits(pred["frame_logits"], target_frame, pos_weight=frame_pos)
    pred_onset = torch.sigmoid(pred["onset_logits"])
    pred_frame = torch.sigmoid(pred["frame_logits"])
    active = (target_frame > 0.1).to(target_velocity.dtype)
    velocity_loss = (torch.abs(pred["velocity"] - target_velocity) * active).sum() / (active.sum() + 1.0)
    pedal_loss = F.l1_loss(pred["pedal"], target_pedal)
    hierarchy_loss = target_pedal.new_tensor(0.0)
    if target_density is not None and target_register is not None:
        density_head_loss = F.l1_loss(pred["density"], target_density)
        register_head_loss = F.binary_cross_entropy(pred["register"], target_register)
        hierarchy_loss = density_head_loss + register_head_loss

    musical_plan_loss = target_pedal.new_tensor(0.0)
    if target_chord is not None and target_bass is not None and target_voicing is not None:
        target_chord = target_chord.to(dtype=pred["chord_logits"].dtype, device=pred["chord_logits"].device)
        target_bass = target_bass.to(dtype=pred["bass_logits"].dtype, device=pred["bass_logits"].device)
        target_voicing = target_voicing.to(dtype=pred["voicing"].dtype, device=pred["voicing"].device)
        chord_loss = F.binary_cross_entropy_with_logits(pred["chord_logits"], target_chord)
        bass_class = torch.argmax(target_bass, dim=1)
        bass_loss = F.cross_entropy(pred["bass_logits"], bass_class)
        voicing_loss = F.l1_loss(pred["voicing"], target_voicing)
        musical_plan_loss = chord_loss + bass_loss + voicing_loss

    event_plan_loss = target_pedal.new_tensor(0.0)
    if target_event is not None and "event" in pred:
        target_event = target_event.to(dtype=pred["event"].dtype, device=pred["event"].device)
        event_plan_loss = F.l1_loss(pred["event"], target_event)

    pc_onset_plan_loss = target_pedal.new_tensor(0.0)
    pc_onset_distribution_loss = target_pedal.new_tensor(0.0)
    pc_onset_f1_loss = target_pedal.new_tensor(0.0)
    if target_pc_onset is not None and "pc_onset_logits" in pred:
        target_pc_onset = target_pc_onset.to(dtype=pred["pc_onset_logits"].dtype, device=pred["pc_onset_logits"].device)
        pc_pos = torch.clamp(
            (target_pc_onset.numel() - target_pc_onset.sum()) / (target_pc_onset.sum() + 1.0),
            1.0,
            60.0,
        )
        pc_onset_plan_loss = F.binary_cross_entropy_with_logits(
            pred["pc_onset_logits"],
            target_pc_onset,
            pos_weight=pc_pos,
        )
        if "pc_onset" in pred:
            pred_pc_onset = torch.clamp(pred["pc_onset"], 0.0, 1.0)
        else:
            pred_pc_onset = torch.sigmoid(pred["pc_onset_logits"])
        target_pc_event = torch.clamp(target_pc_onset, 0.0, 1.0)
        active_pc_frame = (target_pc_event.sum(dim=1, keepdim=True) > 0.1).to(pred_pc_onset.dtype)
        pred_pc_frame_dist = pred_pc_onset / (pred_pc_onset.sum(dim=1, keepdim=True) + 1e-6)
        target_pc_frame_dist = target_pc_event / (target_pc_event.sum(dim=1, keepdim=True) + 1e-6)
        pc_frame_distribution_loss = (
            torch.abs(pred_pc_frame_dist - target_pc_frame_dist) * active_pc_frame
        ).sum() / (active_pc_frame.sum() * 12.0 + 1.0)
        pred_pc_event_usage = pred_pc_onset.sum(dim=2)
        target_pc_event_usage = target_pc_event.sum(dim=2)
        pred_pc_event_usage = pred_pc_event_usage / (pred_pc_event_usage.sum(dim=1, keepdim=True) + 1e-6)
        target_pc_event_usage = target_pc_event_usage / (target_pc_event_usage.sum(dim=1, keepdim=True) + 1e-6)
        pc_event_usage_loss = torch.abs(pred_pc_event_usage - target_pc_event_usage).sum(dim=1).mean()
        pred_pc_event_entropy = -(pred_pc_event_usage * torch.log(pred_pc_event_usage + 1e-6)).sum(dim=1)
        target_pc_event_entropy = -(target_pc_event_usage * torch.log(target_pc_event_usage + 1e-6)).sum(dim=1)
        pc_event_entropy_loss = F.relu(target_pc_event_entropy - pred_pc_event_entropy).mean()
        pred_pc_event_max = pred_pc_event_usage.amax(dim=1)
        target_pc_event_max = target_pc_event_usage.amax(dim=1)
        pc_event_dominance_loss = F.relu(pred_pc_event_max - torch.clamp(target_pc_event_max + 0.05, max=0.45)).mean()
        pc_onset_distribution_loss = (
            pc_frame_distribution_loss
            + pc_event_usage_loss
            + pc_event_entropy_loss
            + pc_event_dominance_loss
        )
        pc_onset_plan_loss = pc_onset_plan_loss + 0.25 * pc_frame_distribution_loss + 0.25 * pc_event_usage_loss
        pooled_pred = F.max_pool1d(
            pred_pc_onset.reshape(-1, 1, pred_pc_onset.shape[-1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(pred_pc_onset)
        pooled_target = F.max_pool1d(
            target_pc_event.reshape(-1, 1, target_pc_event.shape[-1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(target_pc_event)
        soft_precision = (pred_pc_onset * pooled_target).sum(dim=(1, 2)) / (pred_pc_onset.sum(dim=(1, 2)) + 1e-6)
        soft_recall = (target_pc_event * pooled_pred).sum(dim=(1, 2)) / (target_pc_event.sum(dim=(1, 2)) + 1e-6)
        soft_f1 = 2.0 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-6)
        pc_onset_f1_loss = 1.0 - soft_f1.mean()

    role_plan_loss = target_pedal.new_tensor(0.0)
    if target_role is not None and "role" in pred:
        target_role = target_role.to(dtype=pred["role"].dtype, device=pred["role"].device)
        role_plan_loss = F.l1_loss(pred["role"], target_role)

    melody_plan_loss = target_pedal.new_tensor(0.0)
    if target_melody is not None and "melody" in pred:
        target_melody = target_melody.to(dtype=pred["melody"].dtype, device=pred["melody"].device)
        melody_plan_loss = F.l1_loss(pred["melody"], target_melody)

    texture_role_plan_loss = target_pedal.new_tensor(0.0)
    if target_texture_role is not None and "texture_role" in pred:
        target_texture_role = target_texture_role.to(
            dtype=pred["texture_role"].dtype,
            device=pred["texture_role"].device,
        )
        texture_role_plan_loss = F.l1_loss(pred["texture_role"], target_texture_role)

    section_role_plan_loss = target_pedal.new_tensor(0.0)
    if target_section_role is not None and "section_role" in pred:
        target_section_role = target_section_role.to(
            dtype=pred["section_role"].dtype,
            device=pred["section_role"].device,
        )
        section_role_plan_loss = F.l1_loss(pred["section_role"], target_section_role)

    arranger_state_plan_loss = target_pedal.new_tensor(0.0)
    if target_arranger_state is not None and "arranger_state" in pred:
        target_arranger_state = target_arranger_state.to(
            dtype=pred["arranger_state"].dtype,
            device=pred["arranger_state"].device,
        )
        arranger_state_plan_loss = F.l1_loss(pred["arranger_state"], target_arranger_state)

    bass_continuity_plan_loss = target_pedal.new_tensor(0.0)
    if target_bass_continuity is not None and "bass_continuity" in pred:
        target_bass_continuity = target_bass_continuity.to(
            dtype=pred["bass_continuity"].dtype,
            device=pred["bass_continuity"].device,
        )
        bass_continuity_plan_loss = F.l1_loss(pred["bass_continuity"], target_bass_continuity)

    body_melody_state_plan_loss = target_pedal.new_tensor(0.0)
    if target_body_melody_state is not None and "body_melody_state" in pred:
        target_body_melody_state = target_body_melody_state.to(
            dtype=pred["body_melody_state"].dtype,
            device=pred["body_melody_state"].device,
        )
        body_melody_state_plan_loss = F.l1_loss(pred["body_melody_state"], target_body_melody_state)

    section_diversity_plan_loss = target_pedal.new_tensor(0.0)
    if target_section_diversity is not None and "section_diversity" in pred:
        target_section_diversity = target_section_diversity.to(
            dtype=pred["section_diversity"].dtype,
            device=pred["section_diversity"].device,
        )
        section_diversity_plan_loss = F.l1_loss(pred["section_diversity"], target_section_diversity)

    onset_density_loss = F.l1_loss(torch.log1p(pred_onset.sum(dim=1)), torch.log1p(target_onset.sum(dim=1)))
    frame_density_loss = F.l1_loss(torch.log1p(pred_frame.sum(dim=1)), torch.log1p(target_frame.sum(dim=1)))
    density_loss = 0.5 * (onset_density_loss + frame_density_loss)

    key_midi = torch.arange(
        pred_frame.shape[1],
        device=pred_frame.device,
        dtype=pred_frame.dtype,
    ) + float(int(piano_min_midi))
    low_mask = (key_midi <= 52).view(1, -1, 1).to(pred_frame.dtype)
    mid_mask = ((key_midi >= 53) & (key_midi <= 76)).view(1, -1, 1).to(pred_frame.dtype)
    high_mask = (key_midi >= 77).view(1, -1, 1).to(pred_frame.dtype)
    pred_reg_mass = torch.stack(
        [
            (pred_frame * low_mask).sum(dim=1),
            (pred_frame * mid_mask).sum(dim=1),
            (pred_frame * high_mask).sum(dim=1),
        ],
        dim=1,
    )
    target_reg_mass = torch.stack(
        [
            (target_frame * low_mask).sum(dim=1),
            (target_frame * mid_mask).sum(dim=1),
            (target_frame * high_mask).sum(dim=1),
        ],
        dim=1,
    )
    pred_reg_dist = pred_reg_mass / (pred_reg_mass.sum(dim=1, keepdim=True) + 1e-6)
    target_reg_dist = target_reg_mass / (target_reg_mass.sum(dim=1, keepdim=True) + 1e-6)
    texture_active = (target_frame.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
    register_balance_loss = (torch.abs(pred_reg_dist - target_reg_dist) * texture_active).sum() / (
        texture_active.sum() * 3.0 + 1.0
    )
    pred_mid_curve = torch.clamp(pred_reg_mass[:, 1, :] / 8.0, 0.0, 1.0)
    target_mid_curve = torch.clamp(target_reg_mass[:, 1, :] / 8.0, 0.0, 1.0)
    mid_body_loss = (torch.abs(pred_mid_curve - target_mid_curve) * texture_active.squeeze(1)).sum() / (
        texture_active.sum() + 1.0
    )
    texture_balance_loss = register_balance_loss + mid_body_loss + 0.5 * frame_density_loss

    melody_balance_loss = target_pedal.new_tensor(0.0)
    if target_melody is not None:
        target_melody = target_melody.to(dtype=pred_frame.dtype, device=pred_frame.device)
        pred_high_curve = torch.clamp((pred_frame * high_mask).sum(dim=1) / 4.0, 0.0, 1.0)
        pred_upper_curve = torch.clamp((pred_frame * (key_midi >= 72).view(1, -1, 1).to(pred_frame.dtype)).sum(dim=1) / 5.0, 0.0, 1.0)
        target_high_curve = torch.clamp(target_melody[:, 0, :], 0.0, 1.0)
        target_upper_curve = torch.clamp(target_melody[:, 1, :], 0.0, 1.0)
        key_norm_curve = (key_midi - float(int(piano_min_midi))) / max(1.0, float(108 - int(piano_min_midi)))
        pred_pitch_curve = (pred_frame * key_norm_curve.view(1, -1, 1)).sum(dim=1) / (
            pred_frame.sum(dim=1) + 1e-6
        )
        target_pitch_curve = torch.clamp(target_melody[:, 2, :], 0.0, 1.0)
        melody_active = (target_upper_curve > 0.1).to(pred_frame.dtype)
        melody_balance_loss = (
            F.l1_loss(pred_high_curve, target_high_curve)
            + F.l1_loss(pred_upper_curve, target_upper_curve)
            + (torch.abs(pred_pitch_curve - target_pitch_curve) * melody_active).sum() / (melody_active.sum() + 1.0)
        )

    texture_role_balance_loss = target_pedal.new_tensor(0.0)
    if target_texture_role is not None:
        target_texture_role = target_texture_role.to(dtype=pred_frame.dtype, device=pred_frame.device)
        upper_mask = (key_midi >= 72).view(1, -1, 1).to(pred_frame.dtype)
        pred_texture_role = torch.stack(
            [
                torch.clamp((pred_frame * low_mask).sum(dim=1) / 2.0, 0.0, 1.0),
                torch.clamp((pred_frame * mid_mask).sum(dim=1) / 5.0, 0.0, 1.0),
                torch.clamp((pred_onset * mid_mask).sum(dim=1) / 3.0, 0.0, 1.0),
                torch.clamp((pred_frame * upper_mask).sum(dim=1) / 4.0, 0.0, 1.0),
            ],
            dim=1,
        )
        target_texture_role = torch.clamp(target_texture_role, 0.0, 1.0)
        texture_role_active = (target_texture_role.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
        pred_texture_mix = pred_texture_role / (pred_texture_role.sum(dim=1, keepdim=True) + 1e-6)
        target_texture_mix = target_texture_role / (target_texture_role.sum(dim=1, keepdim=True) + 1e-6)
        texture_role_mix_loss = (torch.abs(pred_texture_mix - target_texture_mix) * texture_role_active).sum() / (
            texture_role_active.sum() * 4.0 + 1.0
        )
        pred_texture_activity = torch.clamp(pred_texture_role.sum(dim=1) / 2.0, 0.0, 1.0)
        target_texture_activity = torch.clamp(target_texture_role.sum(dim=1) / 2.0, 0.0, 1.0)
        texture_role_activity_loss = F.l1_loss(pred_texture_activity, target_texture_activity)
        texture_role_balance_loss = texture_role_mix_loss + 0.5 * texture_role_activity_loss

    section_role_balance_loss = target_pedal.new_tensor(0.0)
    if target_section_role is not None:
        target_section_role = target_section_role.to(dtype=pred_frame.dtype, device=pred_frame.device)
        upper_mask = (key_midi >= 72).view(1, -1, 1).to(pred_frame.dtype)
        pred_section_role = torch.stack(
            [
                torch.clamp((pred_frame * low_mask).sum(dim=1) / 2.0, 0.0, 1.0),
                torch.clamp((pred_frame * mid_mask).sum(dim=1) / 5.0, 0.0, 1.0),
                torch.clamp((pred_frame * upper_mask).sum(dim=1) / 4.0, 0.0, 1.0),
                torch.clamp(pred_frame.sum(dim=1) / 8.0, 0.0, 1.0),
            ],
            dim=1,
        )
        target_section_role = torch.clamp(target_section_role, 0.0, 1.0)
        section_active = (target_section_role.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
        section_l1 = (torch.abs(pred_section_role - target_section_role) * section_active).sum() / (
            section_active.sum() * 4.0 + 1.0
        )
        bass_under = (
            F.relu(target_section_role[:, 0, :] - pred_section_role[:, 0, :])
            * (target_section_role[:, 0, :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_section_role[:, 0, :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        section_role_balance_loss = section_l1 + 0.5 * bass_under

    body_melody_state_balance_loss = target_pedal.new_tensor(0.0)
    if target_body_melody_state is not None:
        target_body_melody_state = target_body_melody_state.to(dtype=pred_frame.dtype, device=pred_frame.device)
        upper_mask = (key_midi >= 72).view(1, -1, 1).to(pred_frame.dtype)
        pred_body_melody_state = torch.stack(
            [
                torch.clamp((pred_frame * mid_mask).sum(dim=1) / 5.0, 0.0, 1.0),
                torch.clamp((pred_onset * mid_mask).sum(dim=1) / 3.0, 0.0, 1.0),
                torch.clamp((pred_frame * upper_mask).sum(dim=1) / 4.0, 0.0, 1.0),
                torch.clamp((pred_frame * high_mask).sum(dim=1) / 4.0, 0.0, 1.0),
                torch.clamp((pred_frame * mid_mask).sum(dim=1) / 5.0, 0.0, 1.0),
                torch.clamp((pred_frame * upper_mask).sum(dim=1) / 4.0, 0.0, 1.0),
            ],
            dim=1,
        )
        target_body_melody_state = torch.clamp(target_body_melody_state, 0.0, 1.0)
        state_active = (target_body_melody_state.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
        state_l1 = (
            torch.abs(pred_body_melody_state - target_body_melody_state) * state_active
        ).sum() / (state_active.sum() * 6.0 + 1.0)
        body_under = (
            F.relu(target_body_melody_state[:, [0, 4], :] - pred_body_melody_state[:, [0, 4], :])
            * (target_body_melody_state[:, [0, 4], :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_body_melody_state[:, [0, 4], :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        top_under = (
            F.relu(target_body_melody_state[:, [2, 3, 5], :] - pred_body_melody_state[:, [2, 3, 5], :])
            * (target_body_melody_state[:, [2, 3, 5], :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_body_melody_state[:, [2, 3, 5], :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        inner_under = (
            F.relu(target_body_melody_state[:, 1, :] - pred_body_melody_state[:, 1, :])
            * (target_body_melody_state[:, 1, :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_body_melody_state[:, 1, :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        body_melody_state_balance_loss = 0.25 * state_l1 + body_under + top_under + 0.25 * inner_under

    key_pitch_class = torch.remainder(
        torch.arange(pred_frame.shape[1], device=pred_frame.device, dtype=torch.long) + int(piano_min_midi),
        12,
    )
    pred_chroma = torch.zeros(
        (pred_frame.shape[0], 12, pred_frame.shape[2]),
        dtype=pred_frame.dtype,
        device=pred_frame.device,
    )
    target_chroma = torch.zeros_like(pred_chroma)
    chroma_index = key_pitch_class.view(1, -1, 1).expand(pred_frame.shape[0], -1, pred_frame.shape[2])
    pred_chroma.scatter_add_(1, chroma_index, pred_frame)
    target_chroma.scatter_add_(1, chroma_index, target_frame)
    raw_pred_chroma = pred_chroma
    pred_chroma = pred_chroma / (pred_chroma.sum(dim=1, keepdim=True) + 1e-6)
    target_chroma = target_chroma / (target_chroma.sum(dim=1, keepdim=True) + 1e-6)
    chroma_active = (target_frame.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
    chroma_loss = (torch.abs(pred_chroma - target_chroma) * chroma_active).sum() / (chroma_active.sum() * 12.0 + 1.0)

    pc_onset_alignment_loss = target_pedal.new_tensor(0.0)
    if target_pc_onset is not None:
        target_pc_event = torch.clamp(target_pc_onset.to(dtype=pred_onset.dtype, device=pred_onset.device), 0.0, 1.0)
        pred_pc_onset_from_notes = torch.zeros(
            (pred_onset.shape[0], 12, pred_onset.shape[2]),
            dtype=pred_onset.dtype,
            device=pred_onset.device,
        )
        pred_pc_onset_from_notes.scatter_add_(1, chroma_index, pred_onset)
        pred_pc_onset_from_notes = 1.0 - torch.exp(-torch.clamp(pred_pc_onset_from_notes, min=0.0))
        pred_pc_onset_from_notes = torch.clamp(pred_pc_onset_from_notes, 1e-5, 1.0 - 1e-5)
        pc_note_bce = F.binary_cross_entropy(pred_pc_onset_from_notes, target_pc_event)
        pooled_note_pred = F.max_pool1d(
            pred_pc_onset_from_notes.reshape(-1, 1, pred_pc_onset_from_notes.shape[-1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(pred_pc_onset_from_notes)
        pooled_target = F.max_pool1d(
            target_pc_event.reshape(-1, 1, target_pc_event.shape[-1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(target_pc_event)
        note_precision = (pred_pc_onset_from_notes * pooled_target).sum(dim=(1, 2)) / (
            pred_pc_onset_from_notes.sum(dim=(1, 2)) + 1e-6
        )
        note_recall = (target_pc_event * pooled_note_pred).sum(dim=(1, 2)) / (
            target_pc_event.sum(dim=(1, 2)) + 1e-6
        )
        note_soft_f1 = 2.0 * note_precision * note_recall / (note_precision + note_recall + 1e-6)
        note_f1_loss = 1.0 - note_soft_f1.mean()
        target_pc_active = (target_pc_event.sum(dim=1, keepdim=True) > 0.1).to(pred_onset.dtype)
        pred_note_pc_dist = pred_pc_onset_from_notes / (pred_pc_onset_from_notes.sum(dim=1, keepdim=True) + 1e-6)
        target_pc_event_dist = target_pc_event / (target_pc_event.sum(dim=1, keepdim=True) + 1e-6)
        note_pc_dist_loss = (
            torch.abs(pred_note_pc_dist - target_pc_event_dist) * target_pc_active
        ).sum() / (target_pc_active.sum() * 12.0 + 1.0)
        pc_onset_alignment_loss = pc_note_bce + 0.5 * note_f1_loss + 0.25 * note_pc_dist_loss

    section_diversity_balance_loss = target_pedal.new_tensor(0.0)
    if target_section_diversity is not None:
        target_section_diversity = target_section_diversity.to(dtype=pred_frame.dtype, device=pred_frame.device)
        key_norm_curve = (key_midi - float(int(piano_min_midi))) / max(1.0, float(108 - int(piano_min_midi)))
        pred_pitch_mean = (pred_frame * key_norm_curve.view(1, -1, 1)).sum(dim=1) / (
            pred_frame.sum(dim=1) + 1e-6
        )
        pred_pitch_var = (
            pred_frame
            * torch.square(key_norm_curve.view(1, -1, 1) - pred_pitch_mean[:, None, :])
        ).sum(dim=1) / (pred_frame.sum(dim=1) + 1e-6)
        pred_diversity = torch.stack(
            [
                torch.clamp(pred_frame.sum(dim=1) / 16.0, 0.0, 1.0),
                torch.clamp((1.0 - torch.exp(-raw_pred_chroma)).sum(dim=1) / 12.0, 0.0, 1.0),
                torch.clamp(torch.sqrt(torch.clamp(pred_pitch_var, min=0.0)) * 3.0, 0.0, 1.0),
                torch.clamp(pred_onset.sum(dim=1) / 8.0, 0.0, 1.0),
            ],
            dim=1,
        )
        target_section_diversity = torch.clamp(target_section_diversity, 0.0, 1.0)
        section_diversity_active = (target_section_diversity.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
        section_diversity_under = (
            F.relu(target_section_diversity - pred_diversity) * section_diversity_active
        ).sum() / (section_diversity_active.sum() * 4.0 + 1.0)
        pitch_under = (
            F.relu(target_section_diversity[:, 0, :] - pred_diversity[:, 0, :])
            * (target_section_diversity[:, 0, :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_section_diversity[:, 0, :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        pc_under = (
            F.relu(target_section_diversity[:, 1, :] - pred_diversity[:, 1, :])
            * (target_section_diversity[:, 1, :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_section_diversity[:, 1, :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        range_under = (
            F.relu(target_section_diversity[:, 2, :] - pred_diversity[:, 2, :])
            * (target_section_diversity[:, 2, :] > 0.05).to(pred_frame.dtype)
        ).sum() / ((target_section_diversity[:, 2, :] > 0.05).to(pred_frame.dtype).sum() + 1.0)
        density_under = (
            F.relu(target_section_diversity[:, 3, :] - pred_diversity[:, 3, :])
            * (target_section_diversity[:, 3, :] > 0.02).to(pred_frame.dtype)
        ).sum() / ((target_section_diversity[:, 3, :] > 0.02).to(pred_frame.dtype).sum() + 1.0)
        section_diversity_balance_loss = (
            section_diversity_under
            + 0.5 * pitch_under
            + 0.25 * pc_under
            + 0.25 * range_under
            + 0.1 * density_under
        )

    harmonic_plan_loss = target_pedal.new_tensor(0.0)
    if "harmony" in pred:
        harmonic_plan = torch.clamp(pred["harmony"], 0.0, 1.0)
        harmonic_plan = harmonic_plan / (harmonic_plan.sum(dim=1, keepdim=True) + 1e-6)
        harmonic_plan_loss = (torch.abs(harmonic_plan - target_chroma) * chroma_active).sum() / (
            chroma_active.sum() * 12.0 + 1.0
        )

    source_chroma_loss = target_pedal.new_tensor(0.0)
    if source_chroma is not None:
        src_chroma = torch.clamp(source_chroma.to(dtype=pred_frame.dtype, device=pred_frame.device), 0.0, 1.0)
        src_chroma = src_chroma / (src_chroma.sum(dim=1, keepdim=True) + 1e-6)
        source_active = (src_chroma.sum(dim=1, keepdim=True) > 0.1).to(pred_frame.dtype)
        source_chroma_loss = (torch.abs(pred_chroma - src_chroma) * source_active).sum() / (
            source_active.sum() * 12.0 + 1.0
        )

    pred_pitch_usage = pred_frame.sum(dim=2)
    target_pitch_usage = target_frame.sum(dim=2)
    pred_pitch_usage = pred_pitch_usage / (pred_pitch_usage.sum(dim=1, keepdim=True) + 1e-6)
    target_pitch_usage = target_pitch_usage / (target_pitch_usage.sum(dim=1, keepdim=True) + 1e-6)
    pitch_usage_loss = torch.abs(pred_pitch_usage - target_pitch_usage).sum(dim=1).mean()

    pred_pc_usage = pred_chroma.sum(dim=2)
    target_pc_usage = target_chroma.sum(dim=2)
    pred_pc_usage = pred_pc_usage / (pred_pc_usage.sum(dim=1, keepdim=True) + 1e-6)
    target_pc_usage = target_pc_usage / (target_pc_usage.sum(dim=1, keepdim=True) + 1e-6)
    pred_pitch_entropy = -(pred_pitch_usage * torch.log(pred_pitch_usage + 1e-6)).sum(dim=1)
    target_pitch_entropy = -(target_pitch_usage * torch.log(target_pitch_usage + 1e-6)).sum(dim=1)
    pred_pc_entropy = -(pred_pc_usage * torch.log(pred_pc_usage + 1e-6)).sum(dim=1)
    target_pc_entropy = -(target_pc_usage * torch.log(target_pc_usage + 1e-6)).sum(dim=1)
    pitch_entropy_loss = F.relu(target_pitch_entropy - pred_pitch_entropy).mean()
    pc_entropy_loss = F.relu(target_pc_entropy - pred_pc_entropy).mean()
    pred_max_pitch = pred_pitch_usage.amax(dim=1)
    target_max_pitch = target_pitch_usage.amax(dim=1)
    pred_max_pc = pred_pc_usage.amax(dim=1)
    target_max_pc = target_pc_usage.amax(dim=1)
    pitch_dominance_loss = F.relu(pred_max_pitch - torch.clamp(target_max_pitch + 0.05, max=0.35)).mean()
    pc_dominance_loss = F.relu(pred_max_pc - torch.clamp(target_max_pc + 0.05, max=0.45)).mean()
    anti_collapse_loss = (
        pitch_entropy_loss
        + pc_entropy_loss
        + pitch_dominance_loss
        + pc_dominance_loss
        + pc_onset_distribution_loss
    )

    source_onset_loss = target_pedal.new_tensor(0.0)
    if source_onset is not None:
        source_curve = torch.clamp(source_onset.to(dtype=pred_onset.dtype, device=pred_onset.device), 0.0, 1.0)
        pred_onset_curve = pred_onset.sum(dim=1)
        pred_onset_curve = pred_onset_curve / (pred_onset_curve.amax(dim=1, keepdim=True) + 1e-6)
        density_onset_curve = torch.clamp(pred["density"][:, 0, :], 0.0, 1.0)
        source_onset_loss = 0.5 * (
            F.l1_loss(pred_onset_curve, source_curve) + F.l1_loss(density_onset_curve, source_curve)
        )

    total = (
        onset_loss
        + frame_loss
        + 0.6 * velocity_loss
        + 0.2 * pedal_loss
        + float(density_weight) * density_loss
        + float(chroma_weight) * chroma_loss
        + float(pitch_usage_weight) * pitch_usage_loss
        + float(hierarchy_weight) * hierarchy_loss
        + float(musical_plan_weight) * musical_plan_loss
        + float(event_plan_weight) * event_plan_loss
        + float(pc_onset_plan_weight) * pc_onset_plan_loss
        + float(pc_onset_f1_weight) * pc_onset_f1_loss
        + float(pc_onset_alignment_weight) * pc_onset_alignment_loss
        + float(role_plan_weight) * role_plan_loss
        + float(texture_balance_weight) * texture_balance_loss
        + float(melody_plan_weight) * melody_plan_loss
        + float(melody_balance_weight) * melody_balance_loss
        + float(texture_role_plan_weight) * texture_role_plan_loss
        + float(texture_role_balance_weight) * texture_role_balance_loss
        + float(section_role_plan_weight) * section_role_plan_loss
        + float(section_role_balance_weight) * section_role_balance_loss
        + float(arranger_state_plan_weight) * arranger_state_plan_loss
        + float(bass_continuity_plan_weight) * bass_continuity_plan_loss
        + float(body_melody_state_plan_weight) * body_melody_state_plan_loss
        + float(body_melody_state_balance_weight) * body_melody_state_balance_loss
        + float(section_diversity_plan_weight) * section_diversity_plan_loss
        + float(section_diversity_balance_weight) * section_diversity_balance_loss
        + float(anti_collapse_weight) * anti_collapse_loss
        + float(source_onset_weight) * source_onset_loss
        + float(source_chroma_weight) * source_chroma_loss
        + float(harmonic_plan_weight) * harmonic_plan_loss
    )
    return {
        "loss": total,
        "onset_loss": onset_loss.detach(),
        "frame_loss": frame_loss.detach(),
        "velocity_loss": velocity_loss.detach(),
        "pedal_loss": pedal_loss.detach(),
        "density_loss": density_loss.detach(),
        "chroma_loss": chroma_loss.detach(),
        "pitch_usage_loss": pitch_usage_loss.detach(),
        "hierarchy_loss": hierarchy_loss.detach(),
        "musical_plan_loss": musical_plan_loss.detach(),
        "event_plan_loss": event_plan_loss.detach(),
        "pc_onset_plan_loss": pc_onset_plan_loss.detach(),
        "pc_onset_distribution_loss": pc_onset_distribution_loss.detach(),
        "pc_onset_f1_loss": pc_onset_f1_loss.detach(),
        "pc_onset_alignment_loss": pc_onset_alignment_loss.detach(),
        "role_plan_loss": role_plan_loss.detach(),
        "texture_balance_loss": texture_balance_loss.detach(),
        "melody_plan_loss": melody_plan_loss.detach(),
        "melody_balance_loss": melody_balance_loss.detach(),
        "texture_role_plan_loss": texture_role_plan_loss.detach(),
        "texture_role_balance_loss": texture_role_balance_loss.detach(),
        "section_role_plan_loss": section_role_plan_loss.detach(),
        "section_role_balance_loss": section_role_balance_loss.detach(),
        "arranger_state_plan_loss": arranger_state_plan_loss.detach(),
        "bass_continuity_plan_loss": bass_continuity_plan_loss.detach(),
        "body_melody_state_plan_loss": body_melody_state_plan_loss.detach(),
        "body_melody_state_balance_loss": body_melody_state_balance_loss.detach(),
        "section_diversity_plan_loss": section_diversity_plan_loss.detach(),
        "section_diversity_balance_loss": section_diversity_balance_loss.detach(),
        "anti_collapse_loss": anti_collapse_loss.detach(),
        "source_onset_loss": source_onset_loss.detach(),
        "source_chroma_loss": source_chroma_loss.detach(),
        "harmonic_plan_loss": harmonic_plan_loss.detach(),
    }


__all__ = [
    "PianoRollGenerator",
    "PianoRollModelConfig",
    "ResidualTemporalBlock",
    "piano_roll_loss",
]
