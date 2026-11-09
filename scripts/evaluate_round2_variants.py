"""Locate any training improvement among the existing V5 recipe groups."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import json

import torch

from nsrno.dataset import read_json, write_json
from nsrno.forward_model import load_forward_model
from nsrno.train_forward import evaluate, get_loader

parser = ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
torch.set_num_threads(2)
torch.set_float32_matmul_precision("high")
root = Path("out/v5/forward_cache_v1")
index = read_json(root / "index.json")
checkpoint = torch.load(args.out / "last.pt", map_location="cpu", weights_only=True)
model = load_forward_model(args.out / "last.pt", args.device)
device = torch.device(args.device)
amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
rows = [r for r in index["records"] if r["role"] == "train"]
angles = tuple(range(len(index["acquisition"]["angles_deg"])))
loader_args = SimpleNamespace(angles_per_record=2, batch=4, eval_batch=4, workers=4,
                              seed=20261104, device=args.device)
result = {"checkpoint_epoch": checkpoint["epoch"],
          "base": read_json(args.out / "training_last_all_angles.json")}
for variant in sorted({r["variant"] for r in rows} - {"oa_breast_original", "high_echo_circles_v1"}):
    subset = [r for r in rows if r["variant"] == variant]
    metrics = evaluate(model, get_loader(root, subset, loader_args, fixed=angles),
                       device, amp, angles, checkpoint["scales"], True)
    metrics["records"] = len(subset)
    result[variant] = metrics
    print(json.dumps({"variant": variant, "records": len(subset),
                      "model_loss": metrics["model"]["balanced_iq_loss"],
                      "template_loss": metrics["train_mean"]["balanced_iq_loss"],
                      "rf_error_energy_reduction": metrics["template_error_energy_reduction"]}), flush=True)
write_json(args.out / "training_last_variants_all_angles.json", result)
