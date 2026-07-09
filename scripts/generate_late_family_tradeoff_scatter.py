from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


DATA = [
    {
        "label": "Scratch retrieval",
        "x": 0.4410,
        "y": 0.5872,
        "structure": 0.0754,
        "warble": 0.0090,
        "offset": (14, -18),
        "ha": "left",
    },
    {
        "label": "Sourcehint retrieval",
        "x": 0.3333,
        "y": 0.6296,
        "structure": 0.0997,
        "warble": 0.0113,
        "offset": (10, 18),
        "ha": "left",
    },
    {
        "label": "Pretrained EnCodec",
        "x": 0.3866,
        "y": 0.6250,
        "structure": 0.0473,
        "warble": 0.0246,
        "offset": (12, -6),
        "ha": "left",
    },
    {
        "label": "Retrieval + pretrained",
        "x": 0.4509,
        "y": 0.5909,
        "structure": 0.0697,
        "warble": 0.0095,
        "offset": (20, 6),
        "ha": "left",
    },
    {
        "label": "Scratch structure diff.",
        "x": 0.9943,
        "y": 0.7949,
        "structure": 0.0505,
        "warble": 0.0271,
        "offset": (8, 8),
        "ha": "left",
    },
]


def wrap_label(label: str) -> str:
    mapping = {
        "Scratch retrieval": "Scratch\nretrieval",
        "Sourcehint retrieval": "Sourcehint\nretrieval",
        "Pretrained EnCodec": "Pretrained\nEnCodec",
        "Retrieval + pretrained": "Retrieval +\npretrained",
        "Scratch structure diff.": "Scratch\nstructure diff.",
    }
    return mapping.get(label, label)


def main() -> None:
    base_dir = Path(r"C:\Users\Ahmed\Downloads\project latex\figures")
    out_path = base_dir / "late_family_tradeoff_scatter.pdf"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)

    x = np.array([d["x"] for d in DATA])
    y = np.array([d["y"] for d in DATA])
    structure = np.array([d["structure"] for d in DATA])
    warble = np.array([d["warble"] for d in DATA])

    sizes = 1600 * structure + 80
    norm = Normalize(vmin=float(warble.min()), vmax=float(warble.max()))

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=warble,
        cmap="magma_r",
        norm=norm,
        edgecolors="#2f2f2f",
        linewidths=1.0,
        alpha=0.95,
        zorder=3,
    )

    ax.set_title("Late-family Tradeoff Space")
    ax.set_xlabel("Style confidence")
    ax.set_ylabel("Fullness")
    ax.set_xlim(0.28, 1.08)
    ax.set_ylim(0.54, 0.825)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.285,
        0.815,
        "Bubble size = structure score",
        fontsize=10.5,
        ha="left",
    )

    for item in DATA:
        ax.annotate(
            wrap_label(item["label"]),
            xy=(item["x"], item["y"]),
            xytext=item["offset"],
            textcoords="offset points",
            ha=item["ha"],
            va="center",
            fontsize=9.2,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#444444",
                "lw": 0.85,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=4,
        )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label("Warble (lower is better)")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)


if __name__ == "__main__":
    main()
