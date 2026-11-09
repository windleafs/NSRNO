"""Compare a selected model with correctly paired and deliberately permuted media."""
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
shift = 71
permuted = [{**row, "medium": rows[(i + shift) % len(rows)]["medium"]} for i, row in enumerate(rows)]
settings = read_json(args.out / "run.json")
model = load_forward_model(args.out / "best.pt", "cuda:0")
angles = tuple(range(len(index["acquisition"]["angles_deg"])))
loader_args = SimpleNamespace(angles_per_record=2, batch=4, eval_batch=4, workers=4,
                              seed=20261104, device="cuda:0")
metrics = evaluate(model, get_loader(root, permuted, loader_args, fixed=angles),
                   torch.device("cuda:0"), torch.cuda.is_bf16_supported(),
                   angles, settings["window_scales"], True)
aligned = read_json(args.out / "validation_all_angles.json")
result = {
    "description": "Only the complete c/rho input pair is cyclically permuted within the held-out case; targets and angles retain their original records.",
    "records": len(rows), "shift": shift, "checkpoint_epoch": aligned["best_epoch"],
    "correct_input": aligned, "permuted_input": metrics,
    "relative_rf_error_energy_increase_from_permutation": (
        metrics["model"]["relative_rf_rms"] / aligned["model"]["relative_rf_rms"])**2 - 1,
}
write_json(args.out / "input_pairing.json", result)
print(result["relative_rf_error_energy_increase_from_permutation"], flush=True)
