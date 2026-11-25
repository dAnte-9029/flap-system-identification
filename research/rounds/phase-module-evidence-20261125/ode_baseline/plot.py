"""Plot the complete training and whole-flight development curve."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
curve = json.loads((ROOT / "curve.json").read_text())
result = json.loads((ROOT / "results.json").read_text())
epoch = [r["epoch"] for r in curve]
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
for key,label,color in [("train","Training windows","#4b8b69"),("validation","Development flights","#285177")]:
    ax.plot(epoch, [r[key] for r in curve], color=color, label=label, linewidth=1.5)
ax.set_xlabel("Full training epochs")
ax.set_ylabel("Normalized MSE")
ax.axvline(result["best_epoch"], ls=":", c="#af6527", label="Selected checkpoint")
ax.set_yscale("log")
ax.grid(alpha=.18)
ax.legend(frameon=False)
fig.savefig(ROOT / "training_curve.png", dpi=160)
