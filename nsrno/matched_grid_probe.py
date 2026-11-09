"""Compare matched-shape speed and density RF derivatives on nested grids."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import gc
import json

import numpy as np
from scipy.ndimage import zoom

from .dataset import indexed, read_json
from .fullwave import build_case, initialize, load_generator
from .local_probe import save

SAMPLE = "train_200"
CASE = "Neg_35_Left"
CENTER_XZ_M = (0., .020)
RADIUS_M = .003
WIDTH_M = .00005
STEPS = {"c": (.00025, .0005), "rho": (.005, .01)}
KEYS = {"c": "sound_speed", "rho": "density"}
ANGLES = {1: (0, 5, 10), 2: (5,), 3: (5,)}


def aperture_columns(case):
    return np.flatnonzero((case["x"] >= case["xe"][0]) & (case["x"] <= case["xe"][-1]))


def lift(array, factor):
    factors = [(factor * (n - 1) + 1) / n for n in array.shape]
    return zoom(array, factors, order=1, prefilter=False, grid_mode=False)


def refine(coarse, factor):
    case = coarse.copy()
    case["dx"] = coarse["dx"] / factor
    case["dt"] = coarse["dt"] / factor
    # End-step receiver sample factor-1 is at coarse sample 0's physical time.
    case["nt"] = factor * coarse["nt"]
    case["end"] = (case["nt"] - 1) * case["dt"]
    case["face"] = factor * coarse["face"]
    case["pml"] = factor * coarse["pml"]
    for name in ("x", "z"):
        source = coarse[name]
        fine_indices = np.arange(factor * (source.size - 1) + 1) / factor
        case[name] = np.interp(fine_indices, np.arange(source.size), source)
    case["maps"] = {key: lift(value, factor) for key, value in coarse["maps"].items()}
    for key, original in coarse["maps"].items():
        if not np.array_equal(case["maps"][key][::factor, ::factor], original):
            raise ValueError(f"Refined background failed to preserve coarse nodes: {key}")
    columns = aperture_columns(case)
    case["delays"] = np.zeros(columns.size)
    case["weights"] = np.interp(case["x"][columns], case["xe"], np.hanning(len(case["xe"])))
    coarse_x = coarse["x"][aperture_columns(coarse)]
    if not np.array_equal(case["x"][columns][::factor], coarse_x):
        raise ValueError("Source aperture changed under refinement")
    if not np.isclose(case["weights"].sum() * case["dx"],
                      coarse["weights"].sum() * coarse["dx"], rtol=1e-9, atol=1e-12):
        raise ValueError("Integrated line-source amplitude changed under refinement")
    return case


def boundary(case):
    x, z = CENTER_XZ_M
    distance = np.hypot(case["x"][None, :] - x, case["z"][:, None] - z)
    return (0.5 * (1 - np.tanh((distance - RADIUS_M) / WIDTH_M))).astype(np.float32)


def run_medium(generator, case, angles, factor, row, acq):
    solver = generator.solver_for(case, case["maps"], generator.absorption_model(case))
    pressure = []
    for index in angles:
        angle = acq["angles_deg"][index]
        tref = generator.set_angle(solver, case, angle)
        if abs(tref - row["source_tref_s"][index]) > 1e-12:
            raise ValueError(f"Transmit clock changed at {angle} degrees")
        native, timing = solver.run()
        if factor > 1:
            native = native[factor - 1::factor].copy()
        if native.shape != (acq["native_nt"], acq["elements"]):
            raise ValueError("Invalid common receiver time axis")
        pressure.append(native)
        print(json.dumps({"angle": angle, "dx_um": case["dx"] * 1e6,
                          "seconds": timing["solve_readback_s"]}), flush=True)
    del solver
    gc.collect()
    return pressure


def filter_rf(generator, native, acquisition):
    signed, _ = generator.sim.analytic_channels(
        native, acquisition["native_dt_s"], band=acquisition["band_hz"])
    return signed.T


def metadata(args, digest, case, angles, step):
    return {
        "id": SAMPLE, "case": CASE, "manifest_sha256": digest,
        "factor": args.factor, "dx_m": case["dx"], "dt_s": case["dt"],
        "angle_indices": list(angles), "center_xz_m": list(CENTER_XZ_M),
        "radius_m": RADIUS_M, "width_m": WIDTH_M, "parameter": args.parameter,
        "step": step, "basis_lifted": args.lifted,
    }


def simulate(args, manifest, digest):
    row = indexed(manifest["records"])[SAMPLE]
    if row["case"] != CASE or row["variant"] != "oa_breast_original":
        raise ValueError("Unexpected development sample provenance")
    generator = initialize()
    coarse = build_case(generator, row)
    case = coarse if args.factor == 1 else refine(coarse, args.factor)
    generator.bench.validate_case(case)
    original = case["maps"].copy()
    basis = lift(boundary(coarse), args.factor) if args.lifted else boundary(case)
    angles = ANGLES[args.factor]
    acquisition = manifest["acquisition"]
    args.out.mkdir(parents=True, exist_ok=True)
    steps = STEPS[args.parameter]
    if args.step_index is not None:
        steps = (steps[args.step_index],)
    for step in steps:
        suffix = "_lifted" if args.lifted else ""
        name = f"dx{args.factor}_{args.parameter}_{step:g}{suffix}.npz"
        target = args.out / name
        expected = metadata(args, digest, case, angles, step)
        if args.sign is not None:
            target = signed_path(target, args.sign)
            expected["sign"] = args.sign
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                if json.loads(archive["metadata_json"].item()) != expected:
                    raise ValueError(f"Incompatible prior output {target}")
            continue
        pressures = []
        signs = (-1, 1) if args.sign is None else (args.sign,)
        for sign in signs:
            case["maps"] = {**original, KEYS[args.parameter]:
                            original[KEYS[args.parameter]] * np.exp(sign * step * basis)}
            print(f"{name} sign={sign:+d}", flush=True)
            pressures.append(run_medium(generator, case, angles, args.factor, row, acquisition))
        if args.sign is None:
            save_derivative(generator, target, pressures, step, acquisition, expected)
        else:
            save(target, np.stack(pressures[0]), expected)
            print(f"Saved {target}", flush=True)


def signed_path(target, sign):
    return target.with_name(f"{target.stem}_sign{sign:+d}{target.suffix}")


def save_derivative(generator, target, pressures, step, acquisition, expected):
    derivative = np.stack([
        filter_rf(generator, (plus.astype(np.float64) - minus) / (2 * step), acquisition)
        for minus, plus in zip(*pressures)
    ])
    if (derivative.shape != (len(expected["angle_indices"]), acquisition["elements"],
                             acquisition["rf_samples"]) or not np.isfinite(derivative).all()):
        raise ValueError("Invalid RF derivative")
    save(target, derivative, expected)
    print(f"Saved {target}", flush=True)


def combine(args, manifest, digest):
    step = STEPS[args.parameter][args.step_index]
    target = args.out / f"dx{args.factor}_{args.parameter}_{step:g}.npz"
    generator = load_generator()
    coarse = generator.geometry_case()
    case = {"dx": coarse["dx"] / args.factor, "dt": coarse["dt"] / args.factor}
    expected = metadata(args, digest, case, ANGLES[args.factor], step)
    pressures = []
    for sign in (-1, 1):
        with np.load(signed_path(target, sign), allow_pickle=False) as saved:
            meta = json.loads(saved["metadata_json"].item())
            if meta != {**expected, "sign": sign}:
                raise ValueError(f"Incompatible derivative branch at sign {sign}")
            pressures.append(saved["rf"])
    save_derivative(generator, target, pressures, step, manifest["acquisition"], expected)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=int, choices=tuple(ANGLES))
    parser.add_argument("--parameter", required=True, choices=tuple(STEPS))
    parser.add_argument("--lifted", action="store_true")
    parser.add_argument("--step-index", type=int, choices=(0, 1))
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument("--sign", type=int, choices=(-1, 1))
    branch.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine and (args.step_index is None or args.lifted):
        parser.error("--combine requires --step-index and an analytic boundary")
    if args.lifted and args.factor != 2:
        parser.error("--lifted requires the fine grid")
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    operation = combine if args.combine else simulate
    operation(args, read_json(args.manifest), digest)


if __name__ == "__main__":
    main()
