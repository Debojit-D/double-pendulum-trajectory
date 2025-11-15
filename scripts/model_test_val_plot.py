import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Paths ----------------
OUT_DIR = Path("/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/data/SampleIdeal2/lnn_modelV1")
metrics_path = OUT_DIR / "metrics.json"

with open(metrics_path, "r") as f:
    metrics = json.load(f)

train_loss = metrics["history"]["train_loss"]
val_loss   = metrics["history"]["val_loss"]
epochs = list(range(1, len(train_loss) + 1))

# ---------------- Seaborn style for papers ----------------
sns.set_theme(
    style="darkgrid",   # clean background
    context="paper",     # smaller fonts suited for papers
    font_scale=0.9,      # tweak if you want slightly bigger/smaller text
)

# Typical single-column figure size for two-column papers (inches)
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)

# ---------------- Plot ----------------
ax.plot(epochs, train_loss, label="Train loss", linewidth=1.4)
ax.plot(epochs, val_loss,   label="Validation loss", linewidth=1.4, linestyle="--")

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (normalized space)")
ax.set_yscale("log")

ax.set_title("LNN training on double pendulum")

# Lighten grid but keep it helpful
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)

# Legend inside, but compact
ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    borderpad=0.3,
    handlelength=1.4,
)

# Tight layout for LaTeX import
fig.tight_layout(pad=0.1)

# High-DPI export suitable for LaTeX \includegraphics
out_path = OUT_DIR / "/home/iitgn-robotics/Debojit_WS/double-pendulum-trajectory/scripts/plots/lnn_training_curves_seaborn.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved figure to: {out_path}")

plt.show()
