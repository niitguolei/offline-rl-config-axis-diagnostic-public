"""Generate the three core manuscript figures from committed result records."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmarks"
FIGDIR = ROOT / "paper" / "figures"


def load_json(name: str) -> dict:
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def save_figure(fig, name: str) -> None:
    for suffix in ("png", "pdf", "svg", "tiff"):
        fig.savefig(FIGDIR / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def figure1_dominant_axis() -> None:
    payload = load_json("dominant_axis_r2.json")
    cells = [
        (row["label"].replace("/", "/\n"), float(row["r2"]))
        for row in payload["cells"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = [c[0] for c in cells]
    values = [c[1] for c in cells]
    colors = [
        "#c44e52" if "seed" in label.lower() else "#4c72b0"
        for label in labels
    ]
    ax.bar(range(len(cells)), values, color=colors, width=0.62)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dominant-axis R2")
    ax.set_xticks(range(len(cells)), labels, fontsize=8)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4c72b0"),
            plt.Rectangle((0, 0), 1, 1, color="#c44e52"),
        ],
        labels=["configuration-axis dominant", "seed dominant"],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
    )
    ax.set_title("The free critic is dominated by one factor (axis or seed)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "figure1_dominant_axis")
    plt.close(fig)


def figure2_same_batch() -> None:
    data = load_json("innovation_upgrade.json")["matched_contrast"]
    domains = [("hopper", "IQL/hopper"), ("walker2d", "IQL/walker2d")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    ax = axes[0]
    x = []
    y_critic = []
    y_fqe = []
    offset = 0
    for key, _label in domains:
        per_seed = data[key]["per_seed"]
        for seed in sorted(per_seed):
            critic = per_seed[seed]["critic_corr"]
            fqe = per_seed[seed]["fqe_corr"]
            for i in range(len(critic)):
                x.append(offset + i * 0.18)
                y_critic.append(critic[i])
                y_fqe.append(fqe[i])
            offset += 1.0
    ax.scatter(x, y_critic, s=24, marker="o", color="#4c72b0", label="critic")
    ax.scatter(x, y_fqe, s=24, marker="x", color="#c44e52", label="FQE")
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("Within-slice rank correlation with tau")
    ax.set_xticks([0.72, 1.72], ["Hopper", "Walker2d"], fontsize=7)
    ax.tick_params(axis="x", pad=5)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Per-slice correlations", fontsize=10)

    ax = axes[1]
    means = [
        data["hopper"]["seed_clustered_mean_difference"],
        data["walker2d"]["seed_clustered_mean_difference"],
    ]
    ps = [data["hopper"]["seed_clustered_p"], data["walker2d"]["seed_clustered_p"]]
    ax.bar([0, 1], means, color="#55a868", width=0.55)
    ax.set_ylim(0, 1.85)
    ax.set_ylabel("Critic minus FQE")
    ax.set_xticks([0, 1], ["hopper", "walker2d"])
    for i, (mean, p) in enumerate(zip(means, ps)):
        ax.text(i, mean + 0.04, f"{mean:.3f}\np={p:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Seed-clustered paired endpoint (raw p)", fontsize=10)

    fig.tight_layout()
    save_figure(fig, "figure2_same_batch_contrast")
    plt.close(fig)


def figure3_diagnostic() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.2, 4.4, 2.2, 1.0, "Is the signal a\nfunction of the axis?"),
        (2.8, 4.4, 2.2, 1.0, "Does it track\nquality in-run?"),
        (5.4, 4.4, 2.2, 1.0, "Is it stable across\nits own hyperparameters?"),
        (7.9, 4.4, 1.9, 1.0, "Are there\ndiverged runs?"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#f2f2f2", edgecolor="#333333"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x1, 4.9), (x2, 4.9), arrowstyle="->", mutation_scale=10))

    outcomes = [
        (0.2, 2.5, 2.0, 0.8, "Axis-dependent:\nuse only if sign is known"),
        (2.6, 2.5, 2.0, 0.8, "No quality tracking:\nabstain"),
        (5.0, 2.5, 2.0, 0.8, "Unstable:\nlimit or recalibrate"),
        (7.4, 2.5, 2.4, 0.8, "Diverged runs:\nisolate and gate"),
    ]
    for x, y, w, h, text in outcomes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor="#666666"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
        ax.add_patch(FancyArrowPatch(
            (x + w / 2, 4.4), (x + w / 2, y + h), arrowstyle="->", mutation_scale=10
        ))
    ax.text(5.0, 1.2, "Usable only when all three conditions hold and the arm is stable.",
            ha="center", va="center", fontsize=9)
    ax.set_title("Three-condition diagnostic with a divergence gate", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "figure3_diagnostic")
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure1_dominant_axis()
    figure2_same_batch()
    figure3_diagnostic()
    print(f"wrote figures to {FIGDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
