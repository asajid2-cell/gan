from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_GENRES = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported value: {type(value)!r}")


@dataclass
class HybridCompareConfig:
    output_root: Path = field(default_factory=lambda: Path.home() / "Desktop" / "dggr_hybrid_vocal_compare")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt")
    lab1_checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt")
    source_seconds: float = 60.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    ddim_steps: int = 50
    output_sr: int = 22050
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 328


def picked_songs() -> List[Dict[str, Any]]:
    base = Path.home() / "Downloads"
    songs = [
        {"path": base / "SZA - F2F.flac", "source_genre": "cc0_other"},
        {"path": base / "beabadoobee - fairy song.flac", "source_genre": "cc0_other"},
        {"path": base / "Magdalena Bay - Imaginal Disk - 01-06 Fear, Sex.flac", "source_genre": "cc0_other"},
    ]
    for row in songs:
        if not Path(row["path"]).exists():
            raise FileNotFoundError(f"Missing compare song: {row['path']}")
    return songs


def settings_panel() -> List[Dict[str, Any]]:
    return [
        {
            "label": "fullmix_style",
            "mode": "fullmix",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 2.05,
            "style_strength": 0.74,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.41,
            "source_mel_blend": 0.07,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.16,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "fullmix_dewarble",
            "mode": "fullmix",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 1.95,
            "style_strength": 0.70,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.43,
            "source_mel_blend": 0.08,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.20,
            "hf_start_bin": 54,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hybrid_vocal_style",
            "mode": "hybrid",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 2.00,
            "style_strength": 0.74,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.38,
            "source_mel_blend": 0.04,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.12,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.90,
        },
        {
            "label": "hybrid_vocal_dewarble",
            "mode": "hybrid",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 1.92,
            "style_strength": 0.70,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.40,
            "source_mel_blend": 0.05,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.16,
            "hf_start_bin": 54,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.98,
            "accomp_mix_gain": 0.88,
        },
    ]


def _to_stereo_44k(audio: torch.Tensor, sr: int) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.size(0) == 1:
        audio = audio.repeat(2, 1)
    elif audio.size(0) > 2:
        audio = audio[:2]
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
    return audio


def _to_mono_sr(audio: torch.Tensor, sr_in: int, sr_out: int) -> np.ndarray:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr_in != sr_out:
        audio = torchaudio.functional.resample(audio, sr_in, sr_out)
    return audio.squeeze(0).cpu().numpy().astype(np.float32)


def separate_stems(song: Dict[str, Any], stem_root: Path, source_seconds: float) -> Dict[str, Path]:
    source_path = Path(song["path"])
    stem_dir = stem_root / _slug(source_path.stem)
    source_clip = stem_dir / "source_clip.wav"
    vocal_wav = stem_dir / "vocals.wav"
    accomp_wav = stem_dir / "accompaniment.wav"
    if source_clip.exists() and vocal_wav.exists() and accomp_wav.exists():
        return {"source_clip": source_clip, "vocals": vocal_wav, "accompaniment": accomp_wav}

    stem_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = torchaudio.load(str(source_path))
    audio = _to_stereo_44k(audio, sr)
    max_len = int(round(source_seconds * 44100))
    audio = audio[:, :max_len]
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to("cuda" if torch.cuda.is_available() else "cpu").eval()
    with torch.no_grad():
        est = model(audio.unsqueeze(0).to(next(model.parameters()).device)).cpu()[0]
    sources = list(model.sources)
    vocal_idx = sources.index("vocals")
    vocals = est[vocal_idx]
    accomp = est.sum(dim=0) - vocals

    source_mono = _to_mono_sr(audio, 44100, 22050)
    vocals_mono = _to_mono_sr(vocals, 44100, 22050)
    accomp_mono = _to_mono_sr(accomp, 44100, 22050)

    sf.write(str(source_clip), source_mono, 22050)
    sf.write(str(vocal_wav), vocals_mono, 22050)
    sf.write(str(accomp_wav), accomp_mono, 22050)
    return {"source_clip": source_clip, "vocals": vocal_wav, "accompaniment": accomp_wav}


def run_longform(cfg: HybridCompareConfig, setting: Dict[str, Any], source_audio: Path, source_genre: str, target_genre: str, out_dir: Path, seed: int) -> None:
    generated = out_dir / "longform_coherent.wav"
    if generated.exists():
        return
    cmd = [
        "python",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(cfg.cache_dir),
        "--checkpoint", str(cfg.checkpoint),
        "--lab1-checkpoint", str(cfg.lab1_checkpoint),
        "--source-audio", str(source_audio),
        "--source-genre", str(source_genre),
        "--target-genre", str(target_genre),
        "--source-start-sec", "0.0",
        "--source-seconds", str(cfg.source_seconds),
        "--out-dir", str(out_dir),
        "--chunk-seconds", str(cfg.chunk_seconds),
        "--overlap-seconds", str(cfg.overlap_seconds),
        "--n-frames", str(cfg.n_frames),
        "--ddim-steps", str(cfg.ddim_steps),
        "--assemble-domain", "mel",
        "--device", "auto",
        "--seed", str(seed),
        "--t-start", str(setting["t_start"]),
        "--t-start-end", str(setting["t_start_end"]),
        "--reanchor-every", str(setting["reanchor_every"]),
        "--reanchor-t-start", str(setting["reanchor_t_start"]),
        "--guidance-scale", str(setting["guidance_scale"]),
        "--style-strength", str(setting["style_strength"]),
        "--prefix-blend", str(setting["prefix_blend"]),
        "--source-prefix-blend", str(setting["source_prefix_blend"]),
        "--source-mel-blend", str(setting["source_mel_blend"]),
        "--vocal-source-blend", str(setting["vocal_source_blend"]),
        "--vocal-start-bin", str(setting["vocal_start_bin"]),
        "--vocal-end-bin", str(setting["vocal_end_bin"]),
        "--hf-source-blend", str(setting["hf_source_blend"]),
        "--hf-start-bin", str(setting["hf_start_bin"]),
        "--mel-time-smooth", str(setting["mel_time_smooth"]),
        "--mel-freq-smooth", str(setting["mel_freq_smooth"]),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.log").open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Longform failed for {source_audio} -> {target_genre} [{setting['label']}]")


def make_hybrid_mix(setting: Dict[str, Any], stem_paths: Dict[str, Path], rendered_dir: Path) -> Path:
    vocal_path = stem_paths["vocals"]
    accomp_render = rendered_dir / "longform_coherent.wav"
    hybrid_out = rendered_dir / "hybrid_longform_coherent.wav"
    if hybrid_out.exists():
        return hybrid_out
    vocals, sr_v = sf.read(str(vocal_path), dtype="float32")
    accomp, sr_a = sf.read(str(accomp_render), dtype="float32")
    if sr_v != sr_a:
        raise RuntimeError("Sample-rate mismatch in hybrid remix.")
    n = min(len(vocals), len(accomp))
    vocals = vocals[:n]
    accomp = accomp[:n]
    mix = float(setting.get("accomp_mix_gain", 0.9)) * accomp + float(setting.get("vocal_mix_gain", 0.95)) * vocals
    peak = float(np.max(np.abs(mix))) + 1e-8
    mix = (mix / peak * 0.95).astype(np.float32)
    sf.write(str(hybrid_out), mix, sr_a)
    return hybrid_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline fullmix versus HDemucs vocal-preserve hybrid.")
    parser.add_argument("--out-dir", type=str, default="", help="Existing or desired output directory.")
    args = parser.parse_args()

    cfg = HybridCompareConfig()
    if args.out_dir.strip():
        out_root = Path(args.out_dir)
    else:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = cfg.output_root / f"hybrid_compare_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    songs = picked_songs()
    settings = settings_panel()
    stem_root = out_root / "stems"
    stem_index: Dict[str, Dict[str, Path]] = {}
    for song in songs:
        stem_index[str(song["path"])] = separate_stems(song, stem_root, cfg.source_seconds)

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(row["path"]), "source_genre": row["source_genre"]} for row in songs], indent=2), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    job_idx = 0
    for setting in settings:
        for song in songs:
            stems = stem_index[str(song["path"])]
            render_source = stems["accompaniment"] if setting["mode"] == "hybrid" else stems["source_clip"]
            for target_genre in TARGET_GENRES:
                job_tag = f"{job_idx:03d}_{_slug(Path(song['path']).stem)[:52]}__to__{_slug(target_genre)}"
                out_dir = out_root / "clips" / setting["label"] / job_tag
                run_longform(cfg, setting, render_source, song["source_genre"], target_genre, out_dir, seed=cfg.seed + job_idx)
                final_mix = None
                if setting["mode"] == "hybrid":
                    final_mix = make_hybrid_mix(setting, stems, out_dir)
                manifest_rows.append(
                    {
                        "job_idx": job_idx,
                        "setting_label": setting["label"],
                        "mode": setting["mode"],
                        "source_audio": str(song["path"]),
                        "render_source": str(render_source),
                        "target_genre": target_genre,
                        "output_dir": str(out_dir),
                        "generated_wav": str(out_dir / "longform_coherent.wav"),
                        "final_mix_wav": str(final_mix) if final_mix else str(out_dir / "longform_coherent.wav"),
                    }
                )
                with (out_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(manifest_rows)
                job_idx += 1

    summary = {
        "output_dir": str(out_root),
        "n_songs": len(songs),
        "n_settings": len(settings),
        "target_genres": TARGET_GENRES,
        "total_jobs": len(manifest_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
