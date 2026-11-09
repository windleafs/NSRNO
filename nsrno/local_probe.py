"""Local two-parameter RF sensitivity using paired full-wave perturbations.

The oracle background is the reconstructed archived medium. The only unknowns
are two log-parameter coefficients on one explicitly specified spatial mode.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import numpy as np

from .compare_replay import norm
from .dataset import indexed, read_json, write_json
from .fullwave import build_case, initialize, read_reference, solve

ANGLE_INDICES = (0, 5, 10)
STEP = .01
DIRECTIONS = {
    "c_plus": (STEP, 0), "c_minus": (-STEP, 0),
    "rho_plus": (0, STEP), "rho_minus": (0, -STEP),
    "c_half_plus": (STEP / 2, 0), "c_half_minus": (-STEP / 2, 0),
    "rho_half_plus": (0, STEP / 2), "rho_half_minus": (0, -STEP / 2),
    "mixed": (.008, -.008),
}
MODE = {"x_center_m": 0.0, "z_center_m": .020, "x_sigma_m": .004, "z_sigma_m": .005}


def window(case):
    x = (case["x"] - MODE["x_center_m"]) / MODE["x_sigma_m"]
    z = (case["z"] - MODE["z_center_m"]) / MODE["z_sigma_m"]
    radius_squared = z[:, None] ** 2 + x[None, :] ** 2
    mode = np.exp(-.5 * radius_squared)
    mode *= np.clip((9 - radius_squared) / 3, 0, 1)
    return mode.astype(np.float32)


def save(path, rf, metadata):
    with NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            np.savez(temp, rf=rf, metadata_json=json.dumps(metadata))
            temp.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def perturb(background, mode, c, rho):
    return {
        **background,
        "sound_speed": background["sound_speed"] * np.exp(c * mode),
        "density": background["density"] * np.exp(rho * mode),
    }


def simulate(args, manifest, digest):
    row = indexed(manifest["records"])[args.id]
    if row["case"] not in manifest["development_cases"]:
        raise ValueError("Local pilot is restricted to development cases")
    args.out.mkdir(parents=True, exist_ok=True)
    generator = initialize()
    case = build_case(generator, row)
    reference = read_reference(generator, manifest)
    background = case["maps"].copy()
    mode = window(case)
    for direction in args.directions:
        c, rho = DIRECTIONS[direction]
        output = args.out / (direction + ".npz")
        metadata = {"sample": args.id, "direction": direction, "manifest_sha256": digest,
                    "coefficients_log_c_log_rho": [c, rho], "angle_indices": list(ANGLE_INDICES),
                    "mode": MODE}
        if output.is_file():
            with np.load(output, allow_pickle=False) as result:
                archived_meta = json.loads(result["metadata_json"].item())
            if any(archived_meta.get(key) != value for key, value in metadata.items()):
                raise ValueError(f"Unexpected provenance of {output}")
            continue
        case["maps"] = perturb(background, mode, c, rho)
        rf, trefs, times = solve(generator, case, reference, manifest["acquisition"], ANGLE_INDICES)
        clock_error = float(np.max(np.abs(trefs - np.asarray(row["source_tref_s"])[list(ANGLE_INDICES)])))
        if clock_error > 1e-12:
            raise ValueError("Perturbed simulation changed the observation clock")
        metadata.update(clock_max_abs_s=clock_error, seconds_per_angle=times)
        save(output, rf, metadata)
        print(f"Saved {output}", flush=True)


def remove_gain_and_timing(columns, base, shape):
    """Project away independent infinitesimal gain and time shifts per angle."""
    response = columns.reshape(*shape, -1).copy()
    waveforms = base.reshape(shape)
    for i, waveform in enumerate(waveforms):
        nuisance = np.column_stack((waveform.ravel(),
                                    np.gradient(waveform, axis=-1).ravel()))
        basis, _ = np.linalg.qr(nuisance)
        signals = response[i].reshape(-1, columns.shape[1])
        signals -= basis @ (basis.T @ signals)
    return response.reshape(columns.shape)


def analyze(args, manifest, digest):
    from .dataset import load_observation

    row = indexed(manifest["records"])[args.id]
    ix = list(ANGLE_INDICES)
    original = load_observation(row, manifest).rf[ix].astype(np.float64).ravel()
    base_path = args.replay / (args.id + ".npz")
    base_meta = read_json(base_path.with_suffix(".json"))
    if base_meta["id"] != args.id or base_meta["manifest_sha256"] != digest:
        raise ValueError("Zero-perturbation replay provenance mismatch")
    with np.load(base_path, allow_pickle=False) as saved:
        base = saved["rf"][[base_meta["angle_indices"].index(i) for i in ix]].astype(np.float64).ravel()
    waves = {}
    for key in DIRECTIONS:
        with np.load(args.out / (key + ".npz"), allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].item())
            if (metadata["sample"] != args.id or metadata["manifest_sha256"] != digest
                or metadata["direction"] != key or metadata["mode"] != MODE
                or metadata["angle_indices"] != ix
                or tuple(metadata["coefficients_log_c_log_rho"]) != DIRECTIONS[key]):
                raise ValueError(f"Mismatched simulation for {key}")
            waves[key] = saved["rf"].astype(np.float64).ravel()
    full = np.stack([(waves[key + "_plus"] - waves[key + "_minus"]) / (2 * STEP)
                     for key in ("c", "rho")], axis=1)
    half = np.stack([(waves[key + "_half_plus"] - waves[key + "_half_minus"]) / STEP
                     for key in ("c", "rho")], axis=1)
    singular = np.linalg.svd(half, compute_uv=False)
    product = float(np.dot(half[:, 0], half[:, 1]))
    sensitivity = [norm(col) for col in half.T]
    zero_error = norm(base - original)
    half_response = [norm(half[:, i]) * STEP / 2 for i in range(2)]
    shape = (len(ix), *manifest["rf_shape"][1:])
    adjusted = remove_gain_and_timing(half, base, shape)
    adjusted_singular = np.linalg.svd(adjusted, compute_uv=False)
    truth = np.asarray(DIRECTIONS["mixed"])
    delta = waves["mixed"] - base
    result = {
        "sample": args.id, "case": row["case"], "manifest_sha256": digest,
        "angle_indices": ix, "angles_deg": [manifest["acquisition"]["angles_deg"][i] for i in ix],
        "spatial_mode": MODE, "zero_replay_rf_relative": zero_error / norm(original),
        "zero_error_to_half_step_response": [zero_error / size for size in half_response],
        "half_step_relative_rf": [size / norm(base) for size in half_response],
        "half_vs_full_derivative_relative": [norm(f - h) / norm(h) for f, h in zip(full.T, half.T)],
        "column_cosine": product / np.prod(sensitivity),
        "singular_values_over_rf_norm": (singular / norm(base)).tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "gain_and_timing_projected": {
            "retained_column_norm_fraction": [norm(a) / norm(h) for a, h in zip(adjusted.T, half.T)],
            "singular_values_over_rf_norm": (adjusted_singular / norm(base)).tolist(),
            "condition_number": float(adjusted_singular[0] / adjusted_singular[-1]),
        },
        "mixed_truth": truth.tolist(),
        "mixed_linearization_error": norm(half @ truth - delta) / norm(delta),
        "fits": {},
    }
    for name, columns in (("joint", [0, 1]), ("c_only", [0]), ("rho_only", [1])):
        local = half[:, columns]
        coefficient = np.linalg.lstsq(local, delta, rcond=None)[0]
        parameters = np.zeros(2)
        parameters[columns] = coefficient
        result["fits"][name] = {"log_c_log_rho": parameters.tolist(),
                                "linearized_residual_relative": norm(local @ coefficient - delta) / norm(delta)}
    write_json(args.out / "metrics.json", result)
    print(json.dumps(result, indent=2))


def verify_fit(args, manifest, digest):
    summary = read_json(args.out / "metrics.json")
    ix = list(ANGLE_INDICES)
    if (summary["sample"] != args.id or summary["manifest_sha256"] != digest
        or summary["angle_indices"] != ix or summary["spatial_mode"] != MODE):
        raise ValueError("Local fit provenance mismatch")
    result_path = args.out / (args.fit + "_physical_update.json")
    coefficients = summary["fits"][args.fit]["log_c_log_rho"]
    if np.max(np.abs(coefficients)) > 2 * STEP:
        raise ValueError(f"Fit outside the predeclared local interval: {coefficients}")
    row = indexed(manifest["records"])[args.id]
    generator = initialize()
    case = build_case(generator, row)
    case["maps"] = perturb(case["maps"], window(case), *coefficients)
    reference = read_reference(generator, manifest)
    candidate, trefs, times = solve(generator, case, reference, manifest["acquisition"], ix)
    if not np.allclose(trefs, np.asarray(row["source_tref_s"])[ix], rtol=0, atol=1e-12):
        raise ValueError("Physical verification changed the observation clock")
    with np.load(args.out / "mixed.npz", allow_pickle=False) as saved:
        target = saved["rf"].astype(np.float64)
    base_path = args.replay / (args.id + ".npz")
    base_meta = read_json(base_path.with_suffix(".json"))
    if base_meta["id"] != args.id or base_meta["manifest_sha256"] != digest:
        raise ValueError("Zero-perturbation replay provenance mismatch")
    with np.load(base_path, allow_pickle=False) as saved:
        base = saved["rf"][[base_meta["angle_indices"].index(i) for i in ix]].astype(np.float64)
    result = {
        "sample": args.id, "manifest_sha256": digest, "fit": args.fit,
        "coefficients_log_c_log_rho": coefficients,
        "physical_residual_rms_relative_to_initial": norm(candidate - target) / norm(base - target),
        "seconds_per_angle": times,
    }
    write_json(result_path, result)
    print(json.dumps(result), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("simulate", "analyze", "verify-fit"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--id", default="train_000")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--fit", choices=("joint", "c_only", "rho_only"), default="joint")
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTIONS), default=list(DIRECTIONS))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    if args.command == "simulate":
        simulate(args, manifest, digest)
    else:
        if args.replay is None:
            parser.error("--replay is required for analyze")
        if args.command == "verify-fit":
            verify_fit(args, manifest, digest)
        else:
            analyze(args, manifest, digest)


if __name__ == "__main__":
    main()
