"""Train a fine-scattering correction on the unchanged V5 development split."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import math
import shutil

import numpy as np
import torch

from .dataset import read_json, write_json
from .forward_model import make_forward_model, load_forward_model
from .train_forward import VAL_ANGLES, evaluate, get_loader, save_checkpoint, train_epoch


def mid_score(metrics):
    return float(np.mean(metrics["model"]["window_relative_rf_rms"][1:3]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("out/v5/forward_cache_v1"))
    parser.add_argument("--warm-start", type=Path, default=Path("out/v5/forward_round2_analytic/best.pt"))
    parser.add_argument("--aggregation", choices=("local", "global"), required=True)
    parser.add_argument("--query-coordinate", choices=("nominal", "slowness"), default="nominal")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--angles-per-record", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-eval-records", type=int, default=64)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--plateau-patience", type=int, default=30)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if min(args.epochs, args.batch, args.eval_batch, args.eval_every, args.train_eval_records) < 1:
        parser.error("Training lengths must be positive")
    if not 1 <= args.angles_per_record <= 11:
        parser.error("Invalid number of angles")
    if (args.out / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    index = read_json(args.cache / "index.json")
    train = [r for r in index["records"] if r["role"] == "train"]
    val = [r for r in index["records"] if r["role"] == "validation"]
    if {r["case"] for r in train} & {r["case"] for r in val}:
        raise ValueError("Cases must be disjoint")
    base_train = [r for r in train if r["variant"] in ("oa_breast_original", "high_echo_circles_v1")]
    probe = [base_train[i] for i in np.linspace(0, len(base_train)-1,
             min(args.train_eval_records, len(base_train)), dtype=int)]
    old = torch.load(args.warm_start, map_location="cpu", weights_only=True)
    if old["spec"].get("residual_frame") != "analytic" or "output_window_scales" not in old["spec"]:
        raise ValueError("Warm start must use the existing window-scaled analytic model")
    if tuple(old["spec"]["medium_shape"]) != tuple(index["grid"]["medium_shape"]):
        raise ValueError("Warm-start grid differs from the cache")
    spec = {**old["spec"], "architecture": "fine_scatter_v1",
            "scatter_aggregation": args.aggregation, "scatter_readout": "native"}
    if args.query_coordinate != "nominal":
        spec["scatter_query"] = args.query_coordinate
    scales = spec["output_window_scales"]
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ("resume", "device", "out")}
    if args.query_coordinate == "nominal":
        settings.pop("query_coordinate", None)
    settings.update(warm_start_sha256=sha256(args.warm_start.read_bytes()).hexdigest(),
                    cache_index_sha256=sha256((args.cache / "index.json").read_bytes()).hexdigest(),
                    train_records=len(train), validation_records=len(val),
                    selection="mean_relative_rf_rms_8_to_36_us")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = make_forward_model(spec)
    model.base.load_state_dict(old["model"])
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    val_loader = get_loader(args.cache, val, args, fixed=VAL_ANGLES)
    probe_loader = get_loader(args.cache, probe, args, fixed=VAL_ANGLES)
    if args.resume:
        previous = torch.load(args.out / "last.pt", map_location="cpu", weights_only=True)
        if previous["spec"] != spec or previous["settings"] != settings:
            raise ValueError("Resume requires the same model, data and training configuration")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        history, best, stop_state = previous["history"], previous["best"], previous["stop_state"]
        start = previous["epoch"] + 1
    else:
        initial_val = evaluate(model, val_loader, device, amp, VAL_ANGLES, scales, True)
        initial_train = evaluate(model, probe_loader, device, amp, VAL_ANGLES, scales, True)
        best = mid_score(initial_val)
        stop_state = {"best_training_mid_rms": mid_score(initial_train), "last_improvement_epoch": 0}
        history = [{"epoch": 0, "validation": initial_val, "training_monitor": initial_train, "seconds": 0.}]
        start = 1
        save_checkpoint(args.out / "best.pt", {"spec": spec, "model": model.state_dict(),
                                              "epoch": 0, "validation": initial_val})
        write_json(args.out / "baseline_monitor.json", history[0])
        source = args.out / "source"
        source.mkdir(exist_ok=True)
        for filename in ("forward_model.py", "scatter_model.py", "train_scatter.py",
                         "train_forward.py", "rf_representation.py"):
            shutil.copy2(Path(__file__).parent / filename, source / filename)
    write_json(args.out / "run.json", {**settings, "spec": spec, "window_scales": scales,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "amp_bfloat16": amp, "training_monitor_ids": [r["id"] for r in probe]})
    reason = "epoch_budget"
    for epoch in range(start, args.epochs + 1):
        timer = perf_counter()
        loader = get_loader(args.cache, train, args, epoch-1)
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, loader, optimizer, device, scales, amp, epoch, True)
        row = {"epoch": epoch, "train_loss": loss, "lr": lr}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, amp, VAL_ANGLES, scales, True)
            monitored = evaluate(model, probe_loader, device, amp, VAL_ANGLES, scales, True)
            row.update(validation=metrics, training_monitor=monitored,
                       validation_mid_rms=mid_score(metrics), training_mid_rms=mid_score(monitored))
            if not math.isfinite(row["validation_mid_rms"]) or not math.isfinite(row["training_mid_rms"]):
                raise RuntimeError("Non-finite RF metric")
            if row["validation_mid_rms"] < best:
                best = row["validation_mid_rms"]
                save_checkpoint(args.out / "best.pt", {"spec": spec, "model": model.state_dict(),
                                                      "epoch": epoch, "validation": metrics})
            if row["training_mid_rms"] < stop_state["best_training_mid_rms"] - args.plateau_min_delta:
                stop_state = {"best_training_mid_rms": row["training_mid_rms"],
                              "last_improvement_epoch": epoch}
        scheduler.step()
        row["seconds"] = perf_counter() - timer
        history.append(row)
        save_checkpoint(args.out / "last.pt", {"spec": spec, "settings": settings,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
            "history": history, "stop_state": stop_state, "scales": scales})
        write_json(args.out / "history.json", history)
        print(json.dumps(row), flush=True)
        if ("training_monitor" in row and epoch >= args.min_epochs
                and epoch - stop_state["last_improvement_epoch"] >= args.plateau_patience):
            reason = "training_mid_window_plateau"
            break
    all_angles = tuple(range(len(spec["angles_deg"])))
    last_train = evaluate(model, get_loader(args.cache, base_train, args, fixed=all_angles),
                          device, amp, all_angles, scales, True)
    last_train.update(checkpoint_epoch=history[-1]["epoch"], records=len(base_train))
    write_json(args.out / "training_last_all_angles.json", last_train)
    del model
    model = load_forward_model(args.out / "best.pt", device)
    selected = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)["epoch"]
    for role, rows in (("validation", val), ("training_best", base_train)):
        final = evaluate(model, get_loader(args.cache, rows, args, fixed=all_angles),
                         device, amp, all_angles, scales, True)
        final.update(best_epoch=selected, records=len(rows), mid_window_rms=mid_score(final))
        write_json(args.out / (role + "_all_angles.json"), final)
    write_json(args.out / "completion.json", {"completed_epochs": history[-1]["epoch"],
        "stopping_reason": reason, "best_epoch": selected, "best_validation_mid_rms": best,
        "stop_state": stop_state})
    print(json.dumps(read_json(args.out / "completion.json")), flush=True)


if __name__ == "__main__":
    main()
