"""Report the predeclared 8-36 us comparison of fine-scattering branches."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("/home_nvme/zhuangyang/nsrno/out/v5")
out = root / "forward_round3_summary"
out.mkdir(exist_ok=True)
baseline_dir = root / "forward_round2_analytic"
data = {"baseline": {
    "validation": json.loads((baseline_dir / "validation_all_angles.json").read_text()),
    "training": json.loads((baseline_dir / "training_best_all_angles.json").read_text()),
}}
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for mode, label, color in (("global", "Fine branch + global mixing", "#176ca4"),
                            ("local", "Fine branch + local mixing", "#bd6325"),
                            ("slowness", "Global mixing + slowness query", "#39824d")):
    folder = root / ("forward_round3_" + mode)
    history = json.loads((folder / "history.json").read_text())
    rows = [r for r in history if "validation" in r]
    epochs = [r["epoch"] for r in rows]
    for column, role in enumerate(("training_monitor", "validation")):
        mid = [np.mean(r[role]["model"]["window_relative_rf_rms"][1:3]) for r in rows]
        axes[0, column].plot(epochs, mid, label=label, color=color)
        early = [r[role]["model"]["window_relative_rf_rms"][0] for r in rows]
        axes[1, column].plot(epochs, early, label=label, color=color)
        if mode == "global":
            axes[0, column].axhline(mid[0], color="gray", ls="--", label="Frozen round-2 model")
            axes[1, column].axhline(early[0], color="gray", ls="--", label="Frozen round-2 model")
    result = {key: json.loads((folder / filename).read_text()) for key, filename in (
        ("settings", "run.json"), ("completion", "completion.json"),
        ("validation", "validation_all_angles.json"),
        ("training_best", "training_best_all_angles.json"),
        ("training_last", "training_last_all_angles.json"),
    )}
    result["training_and_monitor_seconds"] = sum(r["seconds"] for r in history)
    result["equal_budget_epoch_60"] = next(r for r in rows if r["epoch"] == 60)
    selected_epoch = result["completion"]["best_epoch"]
    result["selected_monitor"] = next(r for r in rows if r["epoch"] == selected_epoch)
    if (folder / "detail_ablation.json").exists():
        result["detail_ablation"] = json.loads((folder / "detail_ablation.json").read_text())
    data[mode] = result
for column, title in enumerate(("Fixed training subset (64 records)", "Held-out case (270 records)")):
    axes[0, column].set_title(title)
    axes[0, column].set_ylabel("Mean relative RF RMS: 8-36 us")
    axes[1, column].set_ylabel("Relative RF RMS: 0-8 us")
    for row in range(2):
        axes[row, column].set_xlabel("Epoch")
        axes[row, column].grid(alpha=.2)
        axes[row, column].legend(fontsize=8)
fig.suptitle("NSRNO: fine-scale detail and nonlocal receive aggregation")
fig.savefig(out / "learning_curves.png", dpi=160)
(out / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")
for mode in ("global", "local", "slowness"):
    run = data[mode]
    print(json.dumps({"mode": mode, "completion": run["completion"],
        "validation": run["validation"]["model"],
        "training": run["training_best"]["model"]}))
