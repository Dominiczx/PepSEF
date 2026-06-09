
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_ablation_radar(output_png="feature_ablation_radar.png"):
    # Latest data provided by the user
    data = {
        "method": ["None", "PSSM", "HMM", "Both"],
        "score": [0.545332, 0.555536, 0.533401, 0.560050],
        "test_acc": [0.561410, 0.564844, 0.561093, 0.575535],
        "test_pre": [0.604850, 0.608405, 0.613639, 0.616264],
        "test_rec": [0.677523, 0.702897, 0.686572, 0.734147],
        "test_aupoc": [0.457943, 0.452703, 0.441058, 0.456049],
        "test_auc": [0.847211, 0.849191, 0.843987, 0.861265],
    }

    df = pd.DataFrame(data)

    metrics = ["score", "test_acc", "test_pre", "test_rec", "test_aupoc", "test_auc"]
    labels = ["score", "test acc", "test pre", "test rec", "test aupoc", "test auc"]

    # Normalize each axis independently, but keep raw-value ticks on each spoke
    norm_df = df.copy()
    scales = {}

    for m in metrics:
        vals = df[m].to_numpy(dtype=float)
        raw_min = float(vals.min())
        raw_max = float(vals.max())

        # Add a small padding to keep polygons visually separated
        pad = (raw_max - raw_min) * 0.12 if raw_max > raw_min else max(abs(raw_max) * 0.05, 0.01)
        lo = raw_min - pad
        hi = raw_max + pad

        if math.isclose(lo, hi):
            scaled = np.full_like(vals, 0.5, dtype=float)
        else:
            scaled = (vals - lo) / (hi - lo)

        # Keep data away from center for readability
        norm_df[m] = 0.18 + 0.78 * scaled
        scales[m] = {"lo": lo, "hi": hi, "raw_min": raw_min, "raw_max": raw_max}

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(11, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.10)

    # Plot each method
    for _, row in norm_df.iterrows():
        vals = row[metrics].to_numpy(dtype=float)
        vals_closed = np.concatenate([vals, [vals[0]]])
        ax.plot(angles_closed, vals_closed, linewidth=2.2, label=row["method"])
        ax.fill(angles_closed, vals_closed, alpha=0.08)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([])
    ax.grid(True, alpha=0.35)

    # Per-axis raw-value ticks
    tick_fracs = [0.25, 0.50, 0.75, 1.00]
    for ang, m in zip(angles, metrics):
        lo = scales[m]["lo"]
        hi = scales[m]["hi"]

        # spoke line
        ax.plot([ang, ang], [0.18, 1.0], linewidth=1.0, alpha=0.35)

        for frac in tick_fracs:
            r = 0.18 + 0.78 * frac
            raw_val = lo + frac * (hi - lo)
            dtheta = 0.02
            ax.plot([ang - dtheta, ang + dtheta], [r, r], linewidth=0.9, alpha=0.75)

            label_ang = ang + (0.05 if np.cos(ang) >= 0 else -0.05)
            ha = "left" if np.cos(ang) >= 0 else "right"
            ax.text(
                label_ang,
                r,
                f"{raw_val:.3f}",
                fontsize=8.5,
                ha=ha,
                va="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.78, edgecolor="none"),
            )

        # raw min-max range near outer side of each axis
        label_ang = ang + (0.06 if np.cos(ang) >= 0 else -0.06)
        ha = "left" if np.cos(ang) >= 0 else "right"
        ax.text(
            label_ang,
            1.05,
            f"[{scales[m]['raw_min']:.3f}, {scales[m]['raw_max']:.3f}]",
            fontsize=9,
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.9, edgecolor="none"),
        )

    # New requested title
    ax.set_title("Feature Ablation", fontsize=16, pad=20)

    # Move legend closer to the radar chart
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.10, 1.08),
        frameon=True,
        fontsize=10,
        borderaxespad=0.3,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_feature_ablation_radar("feature_ablation_radar.png")
