"""Measure the conditional density sensitivity after fitting speed directions."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .analyze_matched_grid import compare, derivative
from .compare_replay import norm
from .dataset import indexed, load_observation, read_json, write_json
from .density_subspace import (
    ANGLES, DENSITY_STEP, GROUPS, MODES, SAMPLE, VERIFY_ANGLES, branch_path,
)
from .fullwave import load_generator
from .local_probe import remove_gain_and_timing, save
from .matched_grid_probe import WIDTH_M, filter_rf

BANDS = {
    "4-6_MHz": (4., 6.),
    "6-7.5_MHz": (6., 7.5),
    "5-5.5_MHz": (5., 5.5),
    "6.5-7_MHz": (6.5, 7.),
}


def load_column(path, name, digest, factor=2, half=False):
    with np.load(path, allow_pickle=False) as result:
        metadata = json.loads(result["metadata_json"].item())
        expected = {
            "sample": SAMPLE, "manifest_sha256": digest, "factor": factor,
            "angle_indices": list(ANGLES), "name": name, "definition": list(MODES[name]),
            "step": MODES[name][-1] * (0.5 if half else 1.), "width_m": WIDTH_M,
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Mismatched metadata for {name} at {path}")
        return result["rf"][0].astype(np.float64)


def estimate(columns, target):
    matrix = columns.reshape(-1, columns.shape[-1])
    signal = target.ravel()
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0):
        raise ValueError("A speed derivative has zero norm")
    scaled, _, rank, spectrum = np.linalg.lstsq(matrix / scale, signal, rcond=1e-7)
    return scaled / scale, int(rank), float(spectrum[0] / spectrum[-1])


def projection_metrics(columns, target, coefficients, rf):
    predicted = columns @ coefficients
    remainder = norm(target - predicted)
    relative = remainder / norm(target)
    return {
        "remaining_amplitude_fraction": relative,
        "remaining_energy_fraction": relative**2,
        "orthogonal_density_half_percent_over_archived_rf": DENSITY_STEP * remainder / norm(rf),
    }


def project(columns, target, rf, names):
    coefficients, rank, condition = estimate(columns, target)
    result = projection_metrics(columns, target, coefficients, rf)
    result.update(rank=rank, normalized_speed_matrix_condition=condition,
                  log_speed_coefficients_at_half_percent_rho=dict(zip(
                      names, (DENSITY_STEP * coefficients).tolist())))
    return result


def select_band(columns, target, rf, fs, limits):
    size = rf.shape[-1]
    frequency = np.fft.rfftfreq(size, d=1 / fs)
    low, high = limits
    selected = (frequency >= low * 1e6) & (frequency < high * 1e6)

    def filtered(array, axis):
        spectrum = np.fft.rfft(array, axis=axis)
        shape = [1] * array.ndim
        shape[axis] = selected.size
        return np.fft.irfft(spectrum * selected.reshape(shape), n=size, axis=axis)

    return filtered(columns, -2), filtered(target, -1), filtered(rf, -1)


def spectral_equivalence(column, target, fs):
    frequency = np.fft.rfftfreq(target.shape[-1], d=1 / fs)
    use = (frequency >= 4e6) & (frequency < 7.5e6)
    speed = np.fft.rfft(column, axis=-1)[:, use]
    density = np.fft.rfft(target, axis=-1)[:, use]
    den = np.linalg.norm(density)
    inverse_square = (6e6 / frequency[use])**2
    result = {}
    for name, design in (("fixed_coefficient", speed),
                         ("inverse_square_frequency_diagnostic", speed * inverse_square)):
        coefficient = np.vdot(design, density).real / np.vdot(design, design).real
        result[name] = {"remaining_amplitude_fraction": float(np.linalg.norm(density - coefficient * design) / den),
                        "log_speed_coefficient_at_6_MHz_and_half_percent_rho": float(DENSITY_STEP * coefficient)}
    inner = np.sum(np.conj(speed) * density, axis=0).real
    power = np.sum(np.abs(speed)**2, axis=0)
    per_bin = np.divide(inner, power, out=np.zeros_like(inner), where=power > 0)
    result["per_bin_diagnostic"] = {
        "remaining_amplitude_fraction": float(np.linalg.norm(density - per_bin * speed) / den)}
    return result


def analyze(args, manifest, digest):
    row = indexed(manifest["records"])[SAMPLE]
    archived = load_observation(row, manifest).rf[ANGLES[0]].astype(np.float64)
    target = derivative(args.matched, 2, "rho", DENSITY_STEP, digest)[0]
    columns = {"center": derivative(args.matched, 2, "c", MODES["center"][-1], digest)[0]}
    for key in MODES:
        if key == "center":
            continue
        path = args.out / f"dx2_{key}.npz"
        if path.exists():
            columns[key] = load_column(path, key, digest)
        elif not args.partial:
            raise FileNotFoundError(path)
    groups = {**GROUPS, "equivalent_only": ("equivalent",),
              "center_and_equivalent": ("center", "equivalent")}
    result = {"sample": SAMPLE, "case": row["case"], "manifest_sha256": digest,
              "factor": 2, "angle_index": ANGLES[0], "rho_log_step": DENSITY_STEP,
              "mode_definitions": MODES, "groups": {}}
    for label, keys in groups.items():
        if all(key in columns for key in keys):
            matrix = np.stack([columns[key] for key in keys], axis=-1)
            result["groups"][label] = project(matrix, target, archived, keys)
    if "equivalent" in columns:
        half_path = args.out / "dx2_equivalent_half.npz"
        if half_path.exists():
            half = load_column(half_path, "equivalent", digest, half=True)
            result["equivalent_half_step_comparison"] = compare(half, columns["equivalent"])
    fs = manifest["acquisition"]["fs_hz"]
    if "equivalent" in columns:
        result["equivalent_frequency_law"] = spectral_equivalence(columns["equivalent"], target, fs)
    keys = tuple(columns)
    matrix = np.stack(list(columns.values()), axis=-1)
    result["available_modes"] = list(keys)
    result["by_frequency"] = {}
    fitted_by_band = {}
    for name, limits in BANDS.items():
        band = select_band(matrix, target, archived, fs, limits)
        result["by_frequency"][name] = project(*band, keys)
        fitted_by_band[name] = (band, estimate(band[0], band[1])[0])
    low_name, high_name = "4-6_MHz", "6-7.5_MHz"
    low, low_fit = fitted_by_band[low_name]
    high, high_fit = fitted_by_band[high_name]
    result["held_out_frequency"] = {
        "low_fit_evaluated_on_high": projection_metrics(high[0], high[1], low_fit, high[2]),
        "high_fit_evaluated_on_low": projection_metrics(low[0], low[1], high_fit, low[2]),
    }
    train = np.arange(target.shape[0]) % 2 == 0
    held_fit, _, _ = estimate(matrix[train], target[train])
    result["odd_receiver_holdout"] = projection_metrics(
        matrix[~train], target[~train], held_fit, archived[~train])
    joined = np.column_stack((matrix.reshape(-1, len(keys)), target.ravel()))
    adjusted = remove_gain_and_timing(joined, archived.ravel(), (1, *archived.shape))
    adjusted = adjusted.reshape((*archived.shape, len(keys) + 1))
    result["after_gain_and_clock_projection"] = project(
        adjusted[..., :-1], adjusted[..., -1], archived, keys)

    if not args.partial:
        fitted = result["groups"]["with_equivalent"]
        fit = {
            "sample": SAMPLE, "case": row["case"], "manifest_sha256": digest,
            "mode_definitions": {key: MODES[key] for key in keys},
            "log_speed_coefficients": fitted["log_speed_coefficients_at_half_percent_rho"],
            "target_rho_log_step": DENSITY_STEP,
        }
        fit_path = args.out / "fit_fullband.json"
        write_json(fit_path, fit)
        best = np.asarray([fit["log_speed_coefficients"][key] for key in keys])
        explained = matrix @ best
        density_response = DENSITY_STEP * target
        save(args.out / "rf_projection_components.npz",
             np.stack((density_response, explained, density_response - explained)),
             {"sample": SAMPLE, "factor": 2, "angle_index": ANGLES[0],
              "manifest_sha256": digest, "fit_sha256": sha256(fit_path.read_bytes()).hexdigest(),
              "components": ["density_response", "speed_projection", "orthogonal_residual"],
              "axes": ["component", "receiver", "time"], "rho_log_step": DENSITY_STEP})
    name = "metrics_partial.json" if args.partial else "metrics.json"
    write_json(args.out / name, result)
    print(json.dumps(result, indent=2))


def read_transfer(args, manifest, digest, generator, kind, fit_hash):
    angles = VERIFY_ANGLES[args.angle_set]
    waves, metadata = [], []
    for sign in (-1, 1):
        path = branch_path(args.out, args.factor, args.angle_set, kind, sign)
        with np.load(path, allow_pickle=False) as result:
            m = json.loads(result["metadata_json"].item())
            expected = {"sample": SAMPLE, "manifest_sha256": digest, "factor": args.factor,
                        "angle_indices": list(angles), "sign": sign, "kind": kind}
            if kind == "speed":
                expected["fit_sha256"] = fit_hash
            else:
                expected["rho_log_step"] = DENSITY_STEP
            if any(m.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Mismatched verification branch {path}")
            pressure = result["rf"].astype(np.float64)
            shape = (len(angles), manifest["acquisition"]["native_nt"],
                     manifest["acquisition"]["elements"])
            if pressure.shape != shape or not np.isfinite(pressure).all():
                raise ValueError(f"Invalid pressure data in {path}")
            waves.append(pressure)
            metadata.append(m)
    odd = (waves[1] - waves[0]) / 2
    return np.stack([filter_rf(generator, p, manifest["acquisition"]) for p in odd]
                    ).astype(np.float64), metadata


def physical_metrics(target, actual, archived):
    fit_scale = float(np.vdot(actual, target).real / np.vdot(actual, actual).real)
    return {
        "speed_rf_vs_density_rf": compare(target, actual),
        "density_signal_over_archived_rf": norm(target) / norm(archived),
        "measured_residual_over_archived_rf": norm(target - actual) / norm(archived),
        "scalar_recalibration": fit_scale,
        "remaining_amplitude_after_scalar_recalibration": norm(target - fit_scale * actual) / norm(target),
    }


def verify(args, manifest, digest):
    if args.fit is None:
        raise ValueError("Verification requires --fit")
    fit_hash = sha256(args.fit.read_bytes()).hexdigest()
    generator = load_generator()
    actual, metadata = read_transfer(args, manifest, digest, generator, "speed", fit_hash)
    if args.angle_set == "center":
        target = DENSITY_STEP * derivative(args.matched, args.factor, "rho", DENSITY_STEP, digest)
    else:
        target, _ = read_transfer(args, manifest, digest, generator, "density", fit_hash)
    angles = VERIFY_ANGLES[args.angle_set]
    row = indexed(manifest["records"])[SAMPLE]
    archived = load_observation(row, manifest).rf[list(angles)].astype(np.float64)
    degrees = [manifest["acquisition"]["angles_deg"][i] for i in angles]
    result = {
        "factor": args.factor, "manifest_sha256": digest, "fit_sha256": fit_hash,
        "peak_speed_m_s": metadata[0]["peak_speed_m_s"],
        "angles_deg": degrees,
        "aggregate": physical_metrics(target, actual, archived),
        "per_angle": {str(angle): physical_metrics(t, a, r)
                      for angle, t, a, r in zip(degrees, target, actual, archived)},
    }
    result["by_frequency"] = {}
    for name in ("4-6_MHz", "6-7.5_MHz"):
        a, t, r = select_band(actual[..., None], target, archived,
                              manifest["acquisition"]["fs_hz"], BANDS[name])
        result["by_frequency"][name] = physical_metrics(t, a[..., 0], r)
    write_json(args.out / f"verification_dx{args.factor}_{args.angle_set}.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("project", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--factor", type=int, choices=(2, 3), default=3)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--angle-set", choices=tuple(VERIFY_ANGLES), default="center")
    args = parser.parse_args()
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    action = analyze if args.command == "project" else verify
    action(args, read_json(args.manifest), digest)


if __name__ == "__main__":
    main()
