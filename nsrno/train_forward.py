"""Train the first independent NSRNO channel-RF forward operator."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from zlib import crc32
import json
import math
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import read_json, write_json
from .forward_model import ForwardOperator, load_forward_model
from .rf_representation import IQ_SAMPLES, RF_FS, RF_SAMPLES, WINDOW_EDGES_US, decode, time_windows

DURATION_US = 1e6 * RF_SAMPLES / RF_FS
VAL_ANGLES = (0, 5, 10)


class PairedDataset(Dataset):
    def __init__(self, root, records, count, epoch=0, fixed=None):
        self.root, self.records, self.count = root, records, count
        self.epoch, self.fixed = epoch, fixed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        angles = self.fixed
        if angles is None:
            total = len(row["angles_deg"])
            start = (crc32(row["id"].encode()) + self.epoch * self.count) % total
            angles = (start + np.arange(self.count)) % total
        angles = np.asarray(angles, dtype=np.int64)
        medium = np.load(self.root / row["medium"], mmap_mode="c", allow_pickle=False)
        iq = np.load(self.root / row["iq"], mmap_mode="r", allow_pickle=False)[angles]
        return torch.from_numpy(medium), torch.from_numpy(angles), torch.from_numpy(iq)


def training_reference(root, records):
    base = [r for r in records if r["variant"] in ("oa_breast_original", "high_echo_circles_v1")]
    template = np.zeros(base[0]["iq_shape"], dtype=np.float64)
    windows = time_windows(IQ_SAMPLES, DURATION_US)
    energies = np.zeros(len(windows))
    counts = np.zeros(len(windows))
    for row in base:
        iq = np.load(root / row["iq"], mmap_mode="r", allow_pickle=False)
        template += iq
        for i, window in enumerate(windows):
            part = iq[..., window]
            energies[i] += np.sum(np.square(part, dtype=np.float64))
            counts[i] += part.size // 2
    global_scale = float(np.sqrt(energies.sum() / counts.sum()))
    scales = np.maximum(np.sqrt(energies / counts), .025 * global_scale)
    return torch.from_numpy((template / len(base)).astype(np.float32)), global_scale, scales


def get_loader(root, rows, args, epoch=0, fixed=None):
    train = fixed is None
    dataset = PairedDataset(root, rows, args.angles_per_record, epoch, fixed)
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(dataset, batch_size=args.batch if train else args.eval_batch,
                      shuffle=train, num_workers=args.workers,
                      generator=generator, pin_memory=args.device.startswith("cuda"))


def balanced_loss(prediction, target, scales, windows, record_scaled):
    squared = (prediction - target).square().sum(dim=2)
    terms = []
    for window, scale in zip(windows, scales):
        error = squared[..., window].mean(dim=(-2, -1))
        denominator = scale**2
        if record_scaled:
            energy = target[..., window].square().sum(dim=2).mean(dim=(-2, -1))
            denominator = energy.clamp_min(denominator)
        terms.append((error / denominator).mean())
    return torch.stack(terms).mean()


def train_epoch(model, loader, optimizer, device, scales, amp, epoch, record_scaled):
    model.train()
    windows = time_windows(IQ_SAMPLES, DURATION_US)
    total, count = 0., 0
    for step, batch in enumerate(loader, 1):
        medium, angles, target = (array.to(device, non_blocking=True) for array in batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            prediction = model.predict_iq(medium, angles)
        loss = balanced_loss(prediction, target, scales, windows, record_scaled)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}, step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        total += float(loss.detach()) * medium.shape[0]
        count += medium.shape[0]
        if step % 40 == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={total / count:.6f}", flush=True)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, amp, angle_indices, scales=None, record_scaled=False):
    model.eval()
    windows = time_windows(RF_SAMPLES, DURATION_US)
    energy = torch.zeros((3, len(angle_indices), len(windows)), dtype=torch.float64, device=device)
    loss_sums = torch.zeros(2, dtype=torch.float64, device=device)
    record_count = 0
    iq_windows = time_windows(IQ_SAMPLES, DURATION_US)
    template_rf = decode(model.template)[list(angle_indices)]
    for batch in loader:
        medium, angles, target = (array.to(device, non_blocking=True) for array in batch)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            predicted = model.predict_iq(medium, angles)
        if scales is not None:
            loss_sums[0] += balanced_loss(predicted, target, scales, iq_windows, record_scaled) * medium.shape[0]
            loss_sums[1] += balanced_loss(model.template[angles], target, scales, iq_windows, record_scaled) * medium.shape[0]
            record_count += medium.shape[0]
        truth = decode(target)
        prediction = decode(predicted)
        for index, window in enumerate(windows):
            ref = truth[..., window]
            axes = (0, 2, 3)
            energy[0, :, index] += ref.square().sum(dim=axes, dtype=torch.float64)
            energy[1, :, index] += (prediction[..., window] - ref).square().sum(dim=axes, dtype=torch.float64)
            energy[2, :, index] += (template_rf[None, ..., window] - ref).square().sum(dim=axes, dtype=torch.float64)
    target, model_error, mean_error = energy.cpu().numpy()

    def summary(error):
        windows_rms = np.sqrt(error.sum(axis=0) / target.sum(axis=0))
        return {
            "relative_rf_rms": float(np.sqrt(error.sum() / target.sum())),
            "window_relative_rf_rms": windows_rms.tolist(),
            "mean_window_relative_rf_rms": float(windows_rms.mean()),
            "angle_relative_rf_rms": np.sqrt(error.sum(axis=1) / target.sum(axis=1)).tolist(),
        }

    result = {"angles_deg": [model.spec["angles_deg"][i] for i in angle_indices],
              "time_window_edges_us": WINDOW_EDGES_US,
              "model": summary(model_error), "train_mean": summary(mean_error),
              "template_error_energy_reduction": float(1 - model_error.sum() / mean_error.sum()),
              "window_template_error_energy_reduction": (
                  1 - model_error.sum(axis=0) / mean_error.sum(axis=0)).tolist()}
    if scales is not None:
        model_loss, template_loss = (loss_sums / record_count).cpu().tolist()
        result["model"]["balanced_iq_loss"] = model_loss
        result["train_mean"]["balanced_iq_loss"] = template_loss
    return result


def save_checkpoint(path, checkpoint):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(checkpoint, handle)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--angles-per-record", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--loss-scaling", choices=("train-window", "record-window"), default="train-window")
    parser.add_argument("--output-scaling", choices=("global", "train-window"), default="global")
    parser.add_argument("--residual-frame", choices=("baseband", "analytic"), default="baseband")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-eval-records", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--plateau-patience", type=int, default=0)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.eval_batch < 1 or not 1 <= args.angles_per_record <= 11:
        parser.error("Use positive training lengths and 1 to 11 angles per record")
    if args.eval_every < 1 or min(args.train_eval_records, args.min_epochs, args.plateau_patience,
                                  args.plateau_min_delta) < 0:
        parser.error("Invalid evaluation or stopping configuration")
    if args.plateau_patience and not args.train_eval_records:
        parser.error("Plateau stopping requires a fixed training evaluation subset")

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.cache / "index.json"
    index = read_json(index_path)
    digest = sha256(index_path.read_bytes()).hexdigest()
    acq = index["acquisition"]
    train = [r for r in index["records"] if r["role"] == "train"]
    validation = [r for r in index["records"] if r["role"] == "validation"]
    if not train or not validation or {r["case"] for r in train} & {r["case"] for r in validation}:
        raise ValueError("Training and validation must use disjoint cases")
    tref = train[0]["source_tref_s"]
    if any(not np.allclose(r["source_tref_s"], tref, rtol=0, atol=1e-12)
           for r in train + validation):
        raise ValueError("Records do not share the fixed transmit clock")
    spec = {**index["grid"], "receivers": acq["elements"], "pitch_m": acq["pitch_m"],
            "angles_deg": acq["angles_deg"], "source_tref_s": tref}
    if args.residual_frame != "baseband":
        spec["residual_frame"] = args.residual_frame
    settings = {"cache_index_sha256": digest, "epochs": args.epochs, "batch": args.batch,
                "angles_per_record": args.angles_per_record, "seed": args.seed,
                "train_case": index["train_case"], "validation_case": index["validation_case"],
                "train_records": len(train), "validation_records": len(validation)}
    record_scaled = args.loss_scaling == "record-window"
    if record_scaled:
        settings["loss_scaling"] = args.loss_scaling
    if args.residual_frame != "baseband":
        settings["residual_frame"] = args.residual_frame
    extended = (args.residual_frame != "baseband" or args.output_scaling != "global" or args.eval_every != 1
                or args.train_eval_records or args.min_epochs or args.plateau_patience)
    if extended:
        settings.update(output_scaling=args.output_scaling, eval_every=args.eval_every,
                        train_eval_records=args.train_eval_records, min_epochs=args.min_epochs,
                        plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta)
    path = args.out / "last.pt"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if args.resume:
        previous = torch.load(path, map_location="cpu", weights_only=True)
        scales = previous["scales"]
        if args.output_scaling == "train-window":
            spec["output_window_scales"] = scales
        if previous["settings"] != settings or previous["spec"] != spec:
            raise ValueError("Resume requires the same training configuration and cache")
        scales = previous["scales"]
        model = ForwardOperator(spec)
        model.load_state_dict(previous["model"])
        history, best = previous["history"], previous["best"]
        start = previous["epoch"] + 1
    else:
        if path.exists():
            raise FileExistsError("Run already has a checkpoint. Use --resume or another output directory.")
        template, rf_scale, scales = training_reference(args.cache, train)
        scales = scales.tolist()
        if args.output_scaling == "train-window":
            spec["output_window_scales"] = scales
        model = ForwardOperator(spec, template, rf_scale)
        history, best, start = [], math.inf, 1
    stop_state = previous.get("stop_state") if args.resume else None
    if stop_state is None:
        stop_state = {"best_train_loss": math.inf, "last_improvement_epoch": 0}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
    write_json(args.out / "run.json", {**settings, "spec": spec, "amp_bfloat16": amp,
                                     "window_scales": scales, "loss_scaling": args.loss_scaling,
                                     "trainable_parameters": sum(p.numel() for p in model.parameters())})
    code_dir = args.out / "source"
    code_dir.mkdir(exist_ok=True)
    if not args.resume:
        for filename in ("forward_model.py", "train_forward.py", "rf_representation.py"):
            shutil.copy2(Path(__file__).parent / filename, code_dir / filename)
    base_train = [r for r in train if r["variant"] in ("oa_breast_original", "high_echo_circles_v1")]
    selected_count = min(args.train_eval_records, len(base_train))
    train_probe = ([base_train[i] for i in np.linspace(0, len(base_train) - 1, selected_count, dtype=int)]
                   if selected_count else [])
    write_json(args.out / "evaluation_records.json", {
        "training_monitor_ids": [r["id"] for r in train_probe],
        "validation_ids": [r["id"] for r in validation],
        "training_final_ids": [r["id"] for r in base_train],
        "monitor_angles": list(VAL_ANGLES),
    })
    val_loader = get_loader(args.cache, validation, args, fixed=VAL_ANGLES)
    train_probe_loader = get_loader(args.cache, train_probe, args, fixed=VAL_ANGLES) if train_probe else None
    stopping_reason = "epoch_budget"
    for epoch in range(start, args.epochs + 1):
        timer = perf_counter()
        training_loader = get_loader(args.cache, train, args, epoch - 1)
        learning_rate = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, training_loader, optimizer, device, scales, amp, epoch, record_scaled)
        row = {"epoch": epoch, "train_loss": loss, "lr": learning_rate}
        improved = False
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, amp, VAL_ANGLES, scales, record_scaled)
            score = metrics["model"]["mean_window_relative_rf_rms"]
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite validation metric at epoch {epoch}")
            row["validation"] = metrics
            improved = score < best
            if improved:
                best = score
            if train_probe_loader is not None:
                train_metrics = evaluate(model, train_probe_loader, device, amp, VAL_ANGLES, scales, record_scaled)
                row["training_monitor"] = train_metrics
                train_score = train_metrics["model"]["balanced_iq_loss"]
                if train_score < stop_state["best_train_loss"] - args.plateau_min_delta:
                    stop_state = {"best_train_loss": train_score, "last_improvement_epoch": epoch}
        scheduler.step()
        row["seconds"] = perf_counter() - timer
        history.append(row)
        checkpoint = {"spec": spec, "settings": settings, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch, "best": best, "history": history, "scales": scales,
                      "stop_state": stop_state}
        if improved:
            save_checkpoint(args.out / "best.pt", {"spec": spec, "model": model.state_dict(),
                                                  "epoch": epoch, "validation": metrics})
        save_checkpoint(path, checkpoint)
        write_json(args.out / "history.json", history)
        print(json.dumps(row), flush=True)
        if (args.plateau_patience and "training_monitor" in row and epoch >= args.min_epochs
                and epoch - stop_state["last_improvement_epoch"] >= args.plateau_patience):
            stopping_reason = "fixed_training_loss_plateau"
            break

    write_json(args.out / "completion.json", {
        "stopping_reason": stopping_reason, "completed_epochs": history[-1]["epoch"],
        "best_validation_score": best, "stop_state": stop_state,
    })
    best_path = args.out / "best.pt"
    selected = torch.load(best_path, map_location="cpu", weights_only=True)["epoch"]
    all_angles = tuple(range(len(acq["angles_deg"])))
    if extended:
        last_train = evaluate(model, get_loader(args.cache, base_train, args, fixed=all_angles),
                              device, amp, all_angles, scales, record_scaled)
        last_train.update(checkpoint_epoch=history[-1]["epoch"], training_records=len(base_train))
        write_json(args.out / "training_last_all_angles.json", last_train)
    del model
    all_loader = get_loader(args.cache, validation, args, fixed=all_angles)
    selected_model = load_forward_model(best_path, device)
    final = evaluate(selected_model, all_loader, device, amp, all_angles, scales, record_scaled)
    final.update(best_epoch=selected, validation_records=len(validation), train_records=len(train))
    write_json(args.out / "validation_all_angles.json", final)
    if extended:
        train_final = evaluate(selected_model, get_loader(args.cache, base_train, args, fixed=all_angles),
                               device, amp, all_angles, scales, record_scaled)
        train_final.update(checkpoint_epoch=selected, training_records=len(base_train))
        write_json(args.out / "training_best_all_angles.json", train_final)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()

