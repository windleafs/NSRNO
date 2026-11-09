"""Analyze derivative separation and thin-boundary grid sensitivity."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .compare_replay import norm
from .dataset import indexed, load_observation, read_json, write_json
from .local_probe import remove_gain_and_timing
from .matched_grid_probe import ANGLES, CASE, CENTER_XZ_M, RADIUS_M, SAMPLE, STEPS, WIDTH_M

WINDOWS = {
    "full": (0., 60.01),
    "boundary_echo": (18., 34.),
    "later_echo": (34., 52.),
}


def derivative(out, factor, parameter, step, digest, lifted=False):
    suffix = "_lifted" if lifted else ""
    file = out / f"dx{factor}_{parameter}_{step:g}{suffix}.npz"
    with np.load(file, allow_pickle=False) as stored:
        meta = json.loads(stored["metadata_json"].item())
        expected = {
            "id": SAMPLE, "case": CASE, "manifest_sha256": digest, "factor": factor,
            "angle_indices": list(ANGLES[factor]), "center_xz_m": list(CENTER_XZ_M),
            "radius_m": RADIUS_M, "width_m": WIDTH_M, "parameter": parameter,
            "step": step, "basis_lifted": lifted,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected experiment metadata in {file}")
        return stored["rf"].astype(np.float64)


def cosine(first, second):
    value = np.vdot(first, second).real / (norm(first) * norm(second))
    return float(np.clip(value, -1., 1.))


def compare(reference, candidate):
    return {
        "relative_difference": norm(candidate - reference) / norm(reference),
        "cosine": cosine(reference, candidate),
        "norm_candidate_over_reference": norm(candidate) / norm(reference),
    }


def spectral_diagnostic(reference, candidate, fs):
    frequency = np.fft.rfftfreq(reference.shape[-1], d=1 / fs)
    left = np.fft.rfft(reference, axis=-1)
    right = np.fft.rfft(candidate, axis=-1)
    cross = np.sum(np.conj(left) * right, axis=0)
    left_power = np.sum(np.abs(left)**2, axis=0)
    right_power = np.sum(np.abs(right)**2, axis=0)
    result = {}
    for lower, upper in ((4., 5.), (5., 6.), (6., 7.5), (4., 7.5)):
        selected = (frequency >= lower * 1e6) & (frequency < upper * 1e6)
        left_norm = np.sqrt(left_power[selected].sum())
        right_norm = np.sqrt(right_power[selected].sum())
        denom = left_norm * right_norm
        signed_cosine = cross[selected].sum().real / denom
        delays_ns = np.arange(-100, 101)
        phase = np.exp(-2j * np.pi * delays_ns[:, None] * 1e-9 * frequency[selected][None, :])
        shifted = (phase @ cross[selected]).real / denom
        best = int(np.argmax(shifted))
        result[f"{lower:g}-{upper:g}_MHz"] = {
            "cosine": float(signed_cosine),
            "norm_candidate_over_reference": float(right_norm / left_norm),
            "diagnostic_best_candidate_delay_ns": int(delays_ns[best]),
            "cosine_after_diagnostic_delay": float(shifted[best]),
        }
    return result


def separability(c, rho, reference):
    vectors = np.column_stack((c.ravel(), rho.ravel()))
    strength = np.linalg.svd(vectors, compute_uv=False)
    relation = cosine(c, rho)
    scale = norm(reference)
    orthogonal = np.sqrt(max(0., 1 - relation**2))
    return {
        "column_cosine": relation,
        "rho_orthogonal_fraction": float(orthogonal),
        "condition_equal_log_units": float(strength[0] / strength[-1]),
        "c_sensitivity_over_rf_norm": norm(c) / scale,
        "rho_sensitivity_over_rf_norm": norm(rho) / scale,
        "rho_half_percent_orthogonal_over_rf_norm": float(.005 * norm(rho) * orthogonal / scale),
        "singular_values_over_rf_norm": (strength / scale).tolist(),
    }


def frequency_separability(c, rho, rf, fs):
    length = rf.shape[-1]
    frequency = np.fft.rfftfreq(length, d=1 / fs)
    spectra = [np.fft.rfft(array, axis=-1) for array in (c, rho, rf)]
    result = {}
    for low, high in ((4., 6.), (6., 7.5)):
        band = (frequency >= low * 1e6) & (frequency < high * 1e6)
        arrays = [np.fft.irfft(spectrum * band, n=length, axis=-1) for spectrum in spectra]
        result[f"{low:g}-{high:g}_MHz"] = separability(*arrays)
    return result


def grid_summary(factor, rf, entries, time_us, fs):
    shape = rf.shape
    columns = np.column_stack([entries[key][0].ravel() for key in ("c", "rho")])
    adjusted = remove_gain_and_timing(columns, rf.ravel(), shape).reshape((*shape, 2))
    result = {
        "angle_indices": list(ANGLES[factor]),
        "step_stability": {key: compare(*values) for key, values in entries.items()},
        "windows": {},
        "single_angle_full_rf": {},
    }
    for name, (start, end) in WINDOWS.items():
        time = (time_us >= start) & (time_us < end)
        c, rho = (entries[key][0][:, :, time] for key in ("c", "rho"))
        result["windows"][name] = separability(c, rho, rf[:, :, time])
    result["frequency_bands"] = frequency_separability(
        entries["c"][0], entries["rho"][0], rf, fs)
    result["after_per_angle_gain_and_clock_projection"] = separability(
        adjusted[..., 0], adjusted[..., 1], rf)
    for i, angle in enumerate(ANGLES[factor]):
        result["single_angle_full_rf"][str(angle)] = separability(
            entries["c"][0][i], entries["rho"][0][i], rf[i])
    return result


def analyze(args, manifest, digest):
    row = indexed(manifest["records"])[SAMPLE]
    rf = load_observation(row, manifest).rf.astype(np.float64)
    fs = manifest["acquisition"]["fs_hz"]
    time = np.arange(rf.shape[-1]) / fs * 1e6
    report = {"sample": SAMPLE, "case": CASE, "manifest_sha256": digest,
              "angle_deg": manifest["acquisition"]["angles_deg"],
              "center_xz_m": CENTER_XZ_M, "radius_m": RADIUS_M, "width_m": WIDTH_M,
              "log_steps": STEPS, "windows_us": WINDOWS, "grids": {}}
    factors = (1,) if args.coarse_only else (1, 2)
    data = {}
    for factor in factors:
        data[factor] = {key: [derivative(args.out, factor, key, step, digest) for step in steps]
                        for key, steps in STEPS.items()}
        report["grids"][str(factor)] = grid_summary(
            factor, rf[list(ANGLES[factor])], data[factor], time, fs)

    if not args.coarse_only:
        mid = ANGLES[1].index(ANGLES[2][0])
        report["fine_vs_coarse_at_zero_deg"] = {}
        for parameter, steps in STEPS.items():
            report["fine_vs_coarse_at_zero_deg"][parameter] = [
                {"step": step, **compare(fine[0], coarse[mid])}
                for step, coarse, fine in zip(steps, data[1][parameter], data[2][parameter])
            ]
        for name, (start, end) in WINDOWS.items():
            selected = (time >= start) & (time < end)
            report.setdefault("fine_vs_coarse_small_step_by_window", {})[name] = {
                parameter: compare(data[2][parameter][0][0][:, selected],
                                   data[1][parameter][0][mid][:, selected])
                for parameter in STEPS
            }
        if args.lifted_density:
            parameter = "rho"
            step = STEPS[parameter][0]
            lifted = derivative(args.out, 2, parameter, step, digest, lifted=True)[0]
            fine = data[2][parameter][0][0]
            coarse = data[1][parameter][0][mid]
            report["density_discretization_control"] = {
                "fine_analytic_vs_fine_lifted": compare(fine, lifted),
                "fine_lifted_vs_coarse": compare(lifted, coarse),
            }
        report["grid_spectral_diagnostic"] = {
            key: spectral_diagnostic(data[2][key][0][0], data[1][key][0][mid], fs)
            for key in STEPS
        }
        if args.third_grid:
            finest = {key: derivative(args.out, 3, key, steps[0], digest)[0]
                      for key, steps in STEPS.items()}
            report["third_grid"] = {
                "separability": separability(finest["c"], finest["rho"], rf[ANGLES[2][0]]),
                "frequency_bands": frequency_separability(
                    finest["c"], finest["rho"], rf[ANGLES[2][0]], fs),
                "grid_comparisons": {},
            }
            for key, steps in STEPS.items():
                coarse = data[1][key][0][mid]
                fine = data[2][key][0][0]
                new = finest[key]
                scale = norm(rf[ANGLES[2][0]])
                grid_report = {
                    "50_um_vs_16_67_um": compare(new, coarse),
                    "25_um_vs_16_67_um": compare(new, fine),
                    "25_um_vs_16_67_um_spectral": spectral_diagnostic(new, fine, fs),
                    "small_step_response_over_rf_norm": {
                        "50_um": steps[0] * norm(coarse) / scale,
                        "25_um": steps[0] * norm(fine) / scale,
                        "16.67_um": steps[0] * norm(new) / scale,
                    },
                    "25_um_vs_16_67_um_by_window": {},
                }
                for window, (start, end) in WINDOWS.items():
                    selected = (time >= start) & (time < end)
                    grid_report["25_um_vs_16_67_um_by_window"][window] = compare(
                        new[:, selected], fine[:, selected])
                report["third_grid"]["grid_comparisons"][key] = grid_report
    name = "metrics_coarse.json" if args.coarse_only else "metrics.json"
    write_json(args.out / name, report)
    print(json.dumps(report, indent=2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument("--lifted-density", action="store_true")
    parser.add_argument("--third-grid", action="store_true")
    args = parser.parse_args()
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    analyze(args, read_json(args.manifest), digest)


if __name__ == "__main__":
    main()
