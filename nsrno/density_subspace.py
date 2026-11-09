"""Full-wave sound-speed directions for the matched thin-density boundary."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .dataset import indexed, read_json
from .fullwave import build_case, initialize
from .local_probe import save
from .matched_grid_probe import (
    CASE, CENTER_XZ_M, RADIUS_M, SAMPLE, WIDTH_M, filter_rf, refine, run_medium,
)

ANGLES = (5,)
VERIFY_ANGLES = {"center": ANGLES, "heldout": (0, 10)}
DENSITY_STEP = .005
MODES = {
    "center": ("disk", 0., 20., 3., .00025),
    "left": ("disk", -6., 20., 3., .00025),
    "right": ("disk", 6., 20., 3., .00025),
    "shallow": ("disk", 0., 14., 3., .00025),
    "deep": ("disk", 0., 26., 3., .00025),
    "small_disk": ("disk", 0., 20., 1.5, .00025),
    "large_disk": ("disk", 0., 20., 6., .00025),
    "gaussian_2mm": ("gaussian", 0., 20., 2., .00025),
    "gaussian_5mm": ("gaussian", 0., 20., 5., .00025),
    "ring": ("ring", 0., 20., 3., .001),
    "equivalent": ("equivalent", 0., 20., 3., .002),
}
GROUPS = {
    "same_shape": ("center",),
    "positions": ("center", "left", "right", "shallow", "deep"),
    "positions_and_scales": ("center", "left", "right", "shallow", "deep",
                             "small_disk", "large_disk", "gaussian_2mm", "gaussian_5mm"),
}
GROUPS["with_ring"] = (*GROUPS["positions_and_scales"], "ring")
GROUPS["with_equivalent"] = (*GROUPS["with_ring"], "equivalent")


def mode_field(case, name):
    shape, center_x, center_z, size, _ = MODES[name]
    radial = np.hypot(case["x"][None, :] - center_x * 1e-3,
                      case["z"][:, None] - center_z * 1e-3)
    if shape == "gaussian":
        normalized = radial / (size * 1e-3)
        return (np.exp(-0.5 * normalized**2) * np.clip(3. - normalized, 0, 1)).astype(np.float32)
    u = np.tanh((radial - size * 1e-3) / WIDTH_M)
    if shape == "disk":
        return (0.5 * (1 - u)).astype(np.float32)
    if shape == "ring":
        return (1 - u**2).astype(np.float32)
    if shape == "equivalent":
        # Dimensionless negative radial Laplacian of the analytic density mode.
        # The acoustic frequency-domain change of variable gives this term
        # in the locally constant background approximation.
        radial_term = np.divide(0.5 * WIDTH_M, radial, out=np.zeros_like(radial), where=radial > 0)
        return (-2.5 * (1 - u**2) * (u - radial_term)).astype(np.float32)
    raise ValueError(f"Unknown mode {name}")


def paired_derivative(generator, case, original, field, factor, row, acquisition):
    pressures = []
    for sign in (-1, 1):
        case["maps"] = {**original, "sound_speed": original["sound_speed"] * np.exp(sign * field)}
        print(f"sign={sign:+d}", flush=True)
        pressures.append(run_medium(generator, case, ANGLES, factor, row, acquisition)[0])
    difference = (pressures[1].astype(np.float64) - pressures[0]) / 2
    return filter_rf(generator, difference, acquisition)[None]


def columns(args, manifest, digest, generator, case, row):
    acquisition = manifest["acquisition"]
    original = case["maps"].copy()
    for name in args.modes:
        if name == "center":
            raise ValueError("The matched-grid central derivative is reused")
        step = MODES[name][-1] * (0.5 if args.half_step else 1.)
        path = args.out / f"dx{args.factor}_{name}{'_half' if args.half_step else ''}.npz"
        expected = {
            "sample": SAMPLE, "case": CASE, "manifest_sha256": digest,
            "factor": args.factor, "angle_indices": list(ANGLES), "name": name,
            "definition": list(MODES[name]), "step": step, "width_m": WIDTH_M,
        }
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(saved["metadata_json"].item()) != expected:
                    raise ValueError(f"Mismatched existing output {path}")
            continue
        print(f"Computing {path} at log-step {step:g}", flush=True)
        rf = paired_derivative(generator, case, original, step * mode_field(case, name),
                               args.factor, row, acquisition)
        save(path, rf / step, expected)
        print(f"Saved {path}", flush=True)


def branch_path(out, factor, angle_set, kind, sign):
    return out / f"transfer_dx{factor}_{angle_set}_{kind}_sign{sign:+d}.npz"


def verify_branch(args, manifest, digest, generator, case, row):
    angles = VERIFY_ANGLES[args.angle_set]
    meta = {"sample": SAMPLE, "case": CASE, "manifest_sha256": digest,
            "factor": args.factor, "angle_indices": list(angles), "sign": args.sign,
            "kind": args.kind}
    if args.kind == "speed":
        fit = read_json(args.fit)
        if (fit["sample"] != SAMPLE or fit["manifest_sha256"] != digest
            or fit["target_rho_log_step"] != DENSITY_STEP):
            raise ValueError("Fit was computed for another physical observation")
        key = "sound_speed"
        field = np.zeros_like(case["maps"][key])
        for mode, coeff in fit["log_speed_coefficients"].items():
            if fit["mode_definitions"][mode] != list(MODES[mode]):
                raise ValueError(f"Fit used a different definition of {mode}")
            field += coeff * mode_field(case, mode)
        meta.update(
            fit_sha256=sha256(args.fit.read_bytes()).hexdigest(),
            peak_log_c=float(np.max(np.abs(field))),
            peak_speed_m_s=float(np.max(np.abs(case["maps"][key] * np.expm1(field)))),
        )
        if meta["peak_log_c"] > .02:
            raise ValueError("The predicted fit exceeds the predeclared local amplitude bound")
    else:
        key = "density"
        field = DENSITY_STEP * mode_field(case, "center")
        meta["rho_log_step"] = DENSITY_STEP
    path = branch_path(args.out, args.factor, args.angle_set, args.kind, args.sign)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(saved["metadata_json"].item()) != meta:
                raise ValueError(f"Unexpected output at {path}")
        return
    case["maps"][key] *= np.exp(args.sign * field)
    pressure = run_medium(generator, case, angles, args.factor, row, manifest["acquisition"])
    save(path, np.stack(pressure), meta)
    print(f"Saved {path}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("columns", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factor", type=int, choices=(2, 3), default=2)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES))
    parser.add_argument("--half-step", action="store_true")
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--angle-set", choices=tuple(VERIFY_ANGLES), default="center")
    parser.add_argument("--kind", choices=("speed", "density"), default="speed")
    parser.add_argument("--sign", type=int, choices=(-1, 1))
    args = parser.parse_args()
    if args.command == "columns" and not args.modes:
        parser.error("columns requires --modes")
    if args.command == "verify":
        if args.sign is None or (args.kind == "speed" and args.fit is None):
            parser.error("verify requires --sign and, for speed, --fit")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    row = indexed(manifest["records"])[SAMPLE]
    if row["case"] != CASE or row["variant"] != "oa_breast_original":
        raise ValueError("Unexpected case provenance")
    if (CENTER_XZ_M != (MODES["center"][1] * 1e-3, MODES["center"][2] * 1e-3)
        or RADIUS_M != MODES["center"][3] * 1e-3):
        raise ValueError("Matched central geometry has changed")
    generator = initialize()
    case = refine(build_case(generator, row), args.factor)
    generator.bench.validate_case(case)
    operation = columns if args.command == "columns" else verify_branch
    operation(args, manifest, digest, generator, case, row)


if __name__ == "__main__":
    main()
