# plot_training_curves 
# plot_comparison_bar

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "mfcc":    "#00C8FF",
    "logmel":  "#FF6B6B",
    "real":    "#4CAF50",
    "fake":    "#F44336",
    "bg":      "#0D1117",
    "surface": "#161B22",
    "text":    "#E6EDF3",
    "grid":    "#30363D",
}
def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor":   COLORS["surface"],
        "axes.edgecolor":   COLORS["grid"],
        "axes.labelcolor":  COLORS["text"],
        "xtick.color":      COLORS["text"],
        "ytick.color":      COLORS["text"],
        "text.color":       COLORS["text"],
        "grid.color":       COLORS["grid"],
        "grid.alpha":       0.5,
        "font.family":      "monospace",
    })
def plot_feature_comparison(mfcc: np.ndarray,
                             logmel: np.ndarray,
                             label: str = "bonafide",
                             save_path: str = None):
    """Side-by-side visualization of MFCC vs Log-Mel features."""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Feature Comparison — {label.upper()}", fontsize=14, color=COLORS["text"], y=1.02)

    cmap_mfcc   = LinearSegmentedColormap.from_list("mfcc",   ["#0D1117", "#00C8FF", "#FFFFFF"])
    cmap_logmel = LinearSegmentedColormap.from_list("logmel", ["#0D1117", "#FF6B6B", "#FFD700"])

    # MFCC
    im1 = axes[0].imshow(mfcc.squeeze(), aspect='auto', origin='lower', cmap=cmap_mfcc)
    axes[0].set_title("MFCC", color=COLORS["mfcc"], fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("MFCC Coefficients")
    plt.colorbar(im1, ax=axes[0])

    # Log-Mel
    im2 = axes[1].imshow(logmel.squeeze(), aspect='auto', origin='lower', cmap=cmap_logmel)
    axes[1].set_title("Log-Mel Spectrogram", color=COLORS["logmel"], fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Frequency Bins")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, "feature_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_training_curves(history: dict,
                         model_name: str,
                         feature_type: str,
                         save_path: str = None):
    """Plot training loss and accuracy curves."""
    set_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {model_name} | {feature_type.upper()}",
                 color=COLORS["text"], fontsize=13)
    color = COLORS.get(feature_type, "#7FDBFF")
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], color=color, label="Train", linewidth=2)
    val_losses = [m["loss"] for m in history["val_metrics"]]
    ax1.plot(epochs, val_losses, color="#FFD700", label="Val", linewidth=2, linestyle='--')
    ax1.set_title("Loss", color=COLORS["text"])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], color=color, label="Train", linewidth=2)
    val_accs = [m["accuracy"] for m in history["val_metrics"]]
    ax2.plot(epochs, val_accs, color="#FFD700", label="Val", linewidth=2, linestyle='--')
    ax2.set_title("Accuracy (%)", color=COLORS["text"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, f"curves_{model_name}_{feature_type}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_comparison_bar(results_df,
                        metric: str = "eer",
                        save_path: str = None):
    """Bar chart comparing all model × feature combinations on a given metric."""
    import pandas as pd
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    models   = results_df["model"].unique()
    features = results_df["feature"].unique()
    x        = np.arange(len(models))
    width    = 0.35

    for i, feat in enumerate(features):
        vals = [results_df[(results_df["model"]==m) & (results_df["feature"]==feat)][metric].values[0]
                if len(results_df[(results_df["model"]==m) & (results_df["feature"]==feat)]) > 0 else 0
                for m in models]
        color = COLORS.get(feat, "#888")
        bars = ax.bar(x + i*width - width/2, vals, width, label=feat.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=9, color=COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=10)
    lower = "↓ lower is better" if metric == "eer" else "↑ higher is better"
    ax.set_title(f"{metric.upper()} Comparison — MFCC vs Log-Mel ({lower})",
                 color=COLORS["text"], fontsize=13)
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {path}")
    return path

