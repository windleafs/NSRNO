"""Collect completed controlled training runs and plot the observed trajectories."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home_nvme/zhuangyang/nsrno/out/v5")
OUT = ROOT / "forward_round2_summary"
OUT.mkdir(exist_ok=True)
summary = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for name, label, color in (("window", "Window-scaled output", "#176ca4"),
                            ("global", "Global-scaled output", "#bd6325"),
                            ("analytic", "Window scale + analytic frame", "#39824d")):
    folder = ROOT / ("forward_round2_" + name)
    records = json.loads((folder / "history.json").read_text())
    rows = [r for r in records if "training_monitor" in r]
    epochs = [r["epoch"] for r in rows]
    for column, role in enumerate(("training_monitor", "validation")):
        relative_loss = [r[role]["model"]["balanced_iq_loss"] /
                         r[role]["train_mean"]["balanced_iq_loss"] for r in rows]
        gains = [100 * r[role]["template_error_energy_reduction"] for r in rows]
        axes[0, column].plot(epochs, relative_loss, color=color, label=label)
        axes[1, column].plot(epochs, gains, color=color, label=label)
    summary[name] = {
        key: json.loads((folder / file).read_text())
        for key, file in (
            ("settings", "run.json"), ("completion", "completion.json"),
            ("training_best", "training_best_all_angles.json"),
            ("training_last", "training_last_all_angles.json"),
            ("validation_best", "validation_all_angles.json"),
        )
    }
    summary[name]["training_and_monitor_seconds"] = sum(r["seconds"] for r in records)
    variant_file = folder / "training_last_variants_all_angles.json"
    if variant_file.exists():
        summary[name]["training_variants_last"] = json.loads(variant_file.read_text())
for column, title in enumerate(("Fixed training subset (64 records)", "Held-out case (270 records)")):
    axes[0, column].set_title(title)
    axes[0, column].axhline(1, color="gray", lw=1, ls="--")
    axes[0, column].set_ylabel("Balanced loss / template loss")
    axes[1, column].axhline(0, color="gray", lw=1, ls="--")
    axes[1, column].set_ylabel("RF error energy reduction (%)")
    for row in range(2):
        axes[row, column].set_xlabel("Epoch")
        axes[row, column].grid(alpha=.2)
        axes[row, column].legend(fontsize=8)
fig.suptitle("NSRNO: output scaling and complex-frame comparison")
fig.savefig(OUT / "learning_curves.png", dpi=160)
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
for name, run in summary.items():
    print(json.dumps({
        "run": name, "completion": run["completion"],
        "validation_best": run["validation_best"]["model"],
        "training_best": run["training_best"]["model"],
        "training_last": run["training_last"]["model"],
    }))
