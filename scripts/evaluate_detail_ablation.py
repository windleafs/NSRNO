"""Measure the selected model's dependence on its explicit fine-scale channels."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import torch

from nsrno.dataset import read_json, write_json
from nsrno.forward_model import load_forward_model
from nsrno.train_forward import evaluate, get_loader

parser = ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.set_float32_matmul_precision("high")
root = Path("out/v5/forward_cache_v1")
index = read_json(root / "index.json")
rows = [r for r in index["records"] if r["role"] == "validation"]
settings = read_json(args.out / "run.json")
model = load_forward_model(args.out / "best.pt", "cuda:0")
angles = tuple(range(len(index["acquisition"]["angles_deg"])))
loader_args = SimpleNamespace(angles_per_record=2, batch=4, eval_batch=4,
                              workers=4, seed=20261104, device="cuda:0")

def remove_explicit_detail(module, inputs):
    feature = inputs[0].clone()
    feature[:, :32] = 0.
    return (feature,)

hook = model.stem.register_forward_pre_hook(remove_explicit_detail)
try:
    metrics = evaluate(model, get_loader(root, rows, loader_args, fixed=angles),
                       torch.device("cuda:0"), torch.cuda.is_bf16_supported(),
                       angles, settings["window_scales"], True)
finally:
    hook.remove()
full = read_json(args.out / "validation_all_angles.json")
result = {
    "description": "Zero only the 32 packed high-pass c/rho channels before the new stem. Keep the frozen model, smooth background, coordinates, cumulative slowness and input-dependent receive queries.",
    "best_epoch": full["best_epoch"], "records": len(rows),
    "full_model": full, "without_explicit_detail": metrics,
    "window_error_energy_increase": [
        (off/on)**2 - 1 for off, on in zip(
            metrics["model"]["window_relative_rf_rms"],
            full["model"]["window_relative_rf_rms"])],
}
write_json(args.out / "detail_ablation.json", result)
print(result["window_error_energy_increase"], flush=True)
