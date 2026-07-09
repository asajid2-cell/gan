from __future__ import annotations

import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "presentation" / "assets"


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#F7F4EC",
        "figure.facecolor": "#F7F4EC",
    }
)


COLORS = {
    "ink": "#1F2937",
    "muted": "#6B7280",
    "sand": "#F7F4EC",
    "gold": "#C27D38",
    "teal": "#147A73",
    "sage": "#6B8F71",
    "rose": "#C85C5C",
    "slate": "#64748B",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_metrics_dashboard() -> None:
    lab1_leak = load_json(
        REPO_ROOT
        / "saves"
        / "lab1_run_combo_af_gate_exit_v2"
        / "audits_confidence"
        / "leakage_summary.json"
    )
    lab1_gate = load_json(
        REPO_ROOT
        / "saves"
        / "lab1_run_combo_af_gate_exit_v2"
        / "audits_confidence"
        / "gate_summary.json"
    )
    lab2 = load_json(
        REPO_ROOT
        / "saves"
        / "lab2_calibration"
        / "lab2_20260211_015118_lda_cleanup_v2"
        / "validation_summary.json"
    )
    lab3 = load_json(
        REPO_ROOT
        / "saves2"
        / "lab3_codec_transfer"
        / "run1055"
        / "codec_gate_eval.json"
    )

    rows = [
        ("Lab 1 style probe", lab1_leak["style_probe_accuracy"], 0.85, "higher"),
        (
            "Lab 1 leakage",
            lab1_leak["content_leakage_above_baseline"],
            0.15,
            "lower",
        ),
        ("Lab 1 gate AUC", lab1_gate["roc_auc"], 0.90, "higher"),
        ("Lab 2 silhouette", lab2["metrics"]["silhouette"], 0.45, "higher"),
        ("Lab 3 MPS", lab3["metrics"]["mps"], 0.90, "higher"),
        ("Lab 3 style conf.", lab3["metrics"]["style_conf"], 0.85, "higher"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_facecolor(COLORS["sand"])

    y = np.arange(len(rows))
    values = [r[1] for r in rows]
    colors = [
        COLORS["teal"],
        COLORS["rose"],
        COLORS["teal"],
        COLORS["sage"],
        COLORS["gold"],
        COLORS["gold"],
    ]

    ax.barh(y, values, color=colors, height=0.62, edgecolor="none")

    x_max = 1.16
    text_pad = 0.02
    marker_clearance = 0.04

    for idx, (_, value, threshold, direction) in enumerate(rows):
        threshold_x = threshold
        ax.vlines(threshold_x, idx - 0.38, idx + 0.38, color=COLORS["ink"], lw=1.6)
        status = "PASS" if (value >= threshold if direction == "higher" else value <= threshold) else "MISS"
        if direction == "higher":
            delta = value - threshold
            note = f"{value:.3f} vs {threshold:.2f}"
        else:
            delta = threshold - value
            note = f"{value:.3f} vs <= {threshold:.2f}"
        text_x = max(value + text_pad, threshold_x + marker_clearance)
        if text_x > x_max - 0.04:
            text_x = min(value, threshold_x) - text_pad
            ha = "right"
        else:
            ha = "left"
        ax.text(
            max(text_x, 0.03),
            idx,
            f"{status}  {note}",
            va="center",
            ha=ha,
            fontsize=9,
            color=COLORS["ink"],
            fontweight="bold" if delta >= 0 else "normal",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": COLORS["sand"],
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=COLORS["ink"])
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Metric value", color=COLORS["muted"])
    ax.set_title("Measured progress against project gates", loc="left", color=COLORS["ink"], pad=10)
    ax.grid(axis="x", color="#D6D0C4", linewidth=0.8)
    ax.tick_params(axis="x", colors=COLORS["muted"])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()

    fig.text(
        0.125,
        0.93,
        "Threshold markers show the minimum required gate for the demo story.",
        fontsize=9,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=1.2)
    fig.savefig(ASSETS_DIR / "metrics_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def generate_diffusion_curve() -> None:
    history = load_json(
        REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "v2_history.json"
    )
    epochs = np.array([row["epoch"] for row in history])
    train_loss = np.array([row["train_loss"] for row in history])
    val_loss = np.array([row["val_loss"] for row in history])

    best_idx = int(np.argmin(val_loss))
    chosen_epoch = 6
    chosen_idx = int(np.where(epochs == chosen_epoch)[0][0])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(epochs, train_loss, color=COLORS["slate"], lw=2.2, label="train loss")
    ax.plot(epochs, val_loss, color=COLORS["gold"], lw=2.6, label="validation loss")
    ax.scatter(
        epochs[best_idx],
        val_loss[best_idx],
        s=80,
        color=COLORS["teal"],
        zorder=5,
        label=f"best val (epoch {int(epochs[best_idx])})",
    )
    ax.scatter(
        epochs[chosen_idx],
        val_loss[chosen_idx],
        s=80,
        color=COLORS["rose"],
        zorder=5,
        label="best perceived quality (epoch 6)",
    )

    ax.annotate(
        f"lowest val loss\n{val_loss[best_idx]:.4f}",
        xy=(epochs[best_idx], val_loss[best_idx]),
        xytext=(epochs[best_idx] - 4.2, val_loss[best_idx] + 0.018),
        arrowprops={"arrowstyle": "->", "color": COLORS["teal"], "lw": 1.1},
        color=COLORS["teal"],
        fontsize=9,
    )
    ax.annotate(
        f"selected checkpoint\n{val_loss[chosen_idx]:.4f}",
        xy=(epochs[chosen_idx], val_loss[chosen_idx]),
        xytext=(epochs[chosen_idx] + 1.0, val_loss[chosen_idx] + 0.03),
        arrowprops={"arrowstyle": "->", "color": COLORS["rose"], "lw": 1.1},
        color=COLORS["rose"],
        fontsize=9,
    )

    ax.set_title("Diffusion V2 training: numeric optimum vs listening optimum", loc="left", color=COLORS["ink"], pad=14)
    ax.set_xlabel("Epoch", color=COLORS["muted"])
    ax.set_ylabel("Loss", color=COLORS["muted"])
    ax.grid(color="#D6D0C4", linewidth=0.8)
    ax.tick_params(colors=COLORS["muted"])
    ax.legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "diffusion_v2_curve.png", bbox_inches="tight")
    plt.close(fig)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def generate_codec_gallery() -> None:
    files = [
        (
            REPO_ROOT / "examples" / "audio" / "codec_run1055_sample0000_src1_tgt3.wav",
            "CC0/other -> Lo-fi",
        ),
        (
            REPO_ROOT / "examples" / "audio" / "codec_run1055_sample0004_src2_tgt1.wav",
            "Hip-hop -> CC0/other",
        ),
        (
            REPO_ROOT / "examples" / "audio" / "codec_run1055_sample0008_src3_tgt0.wav",
            "Lo-fi -> Baroque",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    for ax, (path, title) in zip(axes, files):
        audio, sr = _load_audio(path)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=96, fmax=sr // 2)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, cmap="magma", ax=ax)
        ax.set_title(title, color=COLORS["ink"], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Best codec run (run1055): example remastered outputs", x=0.05, ha="left", color=COLORS["ink"], fontsize=14)
    fig.savefig(ASSETS_DIR / "codec_gallery.png", bbox_inches="tight")
    plt.close(fig)


def generate_longform_comparison() -> None:
    source_path = (
        REPO_ROOT
        / "saves2"
        / "lab4_longform_coherence"
        / "fullsong_test"
        / "source.wav"
    )
    remaster_path = (
        REPO_ROOT
        / "saves2"
        / "lab4_longform_coherence"
        / "fullsong_test"
        / "longform_coherent.wav"
    )

    src, src_sr = _load_audio(source_path)
    gen, gen_sr = _load_audio(remaster_path)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 4.8), constrained_layout=True)
    for ax, audio, sr, title in [
        (axes[0], src, src_sr, "Source track"),
        (axes[1], gen, gen_sr, "160s long-form remaster"),
    ]:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            hop_length=2048,
            fmax=sr // 2,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=2048, x_axis="time", y_axis=None, cmap="viridis", ax=ax)
        ax.set_title(title, color=COLORS["ink"], fontsize=11, loc="left")
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[1].set_xlabel("Seconds", color=COLORS["muted"])
    fig.suptitle("Long-form coherence test", x=0.05, ha="left", color=COLORS["ink"], fontsize=14)
    fig.savefig(ASSETS_DIR / "longform_comparison.png", bbox_inches="tight")
    plt.close(fig)


def generate_demo_audio_grid() -> None:
    files = [
        (
            REPO_ROOT / "examples" / "audio" / "codec_run1055_sample0000_src1_tgt3.wav",
            "Codec: CC0/other -> Lo-fi",
        ),
        (
            REPO_ROOT / "examples" / "audio" / "codec_run1055_sample0008_src3_tgt0.wav",
            "Codec: Lo-fi -> Baroque",
        ),
        (
            REPO_ROOT / "examples" / "audio" / "diffusion_v2_run_d002_epoch006_00_gen.wav",
            "Diffusion V2 sample",
        ),
        (
            REPO_ROOT
            / "saves2"
            / "lab4_longform_coherence"
            / "fullsong_test"
            / "longform_coherent.wav",
            "Long-form remaster",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.6), constrained_layout=True)
    axes = axes.flatten()

    for ax, (path, title) in zip(axes, files):
        audio, sr = _load_audio(path)
        duration = len(audio) / sr
        if duration > 12.0:
            start = int(sr * 30.0)
            end = min(len(audio), start + int(sr * 12.0))
            audio = audio[start:end]
        times = np.linspace(0.0, len(audio) / sr, num=len(audio))
        ax.plot(times, audio, color=COLORS["teal"], linewidth=0.7)
        ax.fill_between(times, audio, 0.0, color=COLORS["gold"], alpha=0.22)
        ax.set_title(title, loc="left", fontsize=11, color=COLORS["ink"])
        ax.set_xlim(times[0], times[-1] if len(times) else 1.0)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=8)
        ax.set_xlabel("seconds", color=COLORS["muted"], fontsize=8)
        ax.grid(axis="x", color="#D6D0C4", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Demo clip thumbnails used in the lecture deck", x=0.05, ha="left", fontsize=14, color=COLORS["ink"])
    fig.savefig(ASSETS_DIR / "demo_audio_grid.png", bbox_inches="tight")
    plt.close(fig)


def generate_longform_excerpt_comparison() -> None:
    source_path = (
        REPO_ROOT
        / "saves2"
        / "lab4_longform_coherence"
        / "fullsong_test"
        / "source.wav"
    )
    remaster_path = (
        REPO_ROOT
        / "saves2"
        / "lab4_longform_coherence"
        / "fullsong_test"
        / "longform_coherent.wav"
    )
    start_sec = 30
    duration_sec = 20

    src, src_sr = _load_audio(source_path)
    gen, gen_sr = _load_audio(remaster_path)

    src = src[int(start_sec * src_sr): int((start_sec + duration_sec) * src_sr)]
    gen = gen[int(start_sec * gen_sr): int((start_sec + duration_sec) * gen_sr)]

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, audio, sr, title in [
        (axes[0], src, src_sr, "Source excerpt (30s-50s)"),
        (axes[1], gen, gen_sr, "Long-form remaster excerpt (30s-50s)"),
    ]:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            hop_length=1024,
            fmax=sr // 2,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=1024, x_axis="time", y_axis=None, cmap="magma", ax=ax)
        ax.set_title(title, loc="left", fontsize=11, color=COLORS["ink"])
        ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[1].set_xlabel("Seconds in excerpt", color=COLORS["muted"])
    fig.suptitle("Matched excerpt for long-form listening comparison", x=0.05, ha="left", fontsize=14, color=COLORS["ink"])
    fig.savefig(ASSETS_DIR / "longform_excerpt_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_assets_dir()
    generate_metrics_dashboard()
    generate_diffusion_curve()
    generate_codec_gallery()
    generate_longform_comparison()
    generate_demo_audio_grid()
    generate_longform_excerpt_comparison()
    print(f"Saved figures to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
