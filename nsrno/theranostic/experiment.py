"""Controlled acquisition comparison without archived data or trained models."""

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

from .inverse import conditional_sensitivity, solve_inverse
from .setup import make_basis, make_geometry, make_grid_axes, make_phantom, physical_maps, validate_config
from .wave import Grid, Solver


def scenario_apertures(config):
    """Only reciprocal ports supporting the transmitted band are observed.

    With no LF side reception, the disjoint HF centre cannot measure the LF
    echo. Those two hardware scenarios therefore reduce to central HF data.
    The virtual central dual-band comparison remains an idealized reference.
    """
    side = config["geometry"]["side_receive"]
    return {
        "central_hf": ["center_hf"],
        "central_dual": ["center_hf", "center_lf"],
        "flat_dual": ["center_hf", "flat_lf"] if side else ["center_hf"],
        "arc_dual": ["center_hf", "arc_lf"] if side else ["center_hf"],
    }


def holdout_mask(count, stride):
    """Hold out every stride-th port on both Tx and Rx (no reciprocal leak)."""
    if count < 2 or stride < 2 or stride > count:
        raise ValueError("holdout_stride must be between 2 and each aperture's element count")
    withheld = np.arange(count) % stride == stride - 1
    return (withheld[:, None] | withheld[None, :]).ravel()


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _relative_error(prediction, target):
    denominator = np.linalg.norm(target)
    if denominator < 1e-30:
        return None
    return float(100 * np.linalg.norm(prediction - target) / denominator)


def _rms(values):
    return float(np.sqrt(np.mean(np.abs(values)**2)))


def _source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def _provenance(phantom_path):
    try:
        repository = Path(__file__).resolve().parents[2]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repository,
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "git_revision": revision, "git_worktree_dirty": dirty, "code_sha256": _source_hashes(),
            "phantom_path": str(Path(phantom_path).resolve()) if phantom_path else None,
            "phantom_sha256": hashlib.sha256(Path(phantom_path).read_bytes()).hexdigest() if phantom_path else None}


@dataclass
class Observation:
    aperture: str
    frequency_mhz: float
    jacobian: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    nodal_born: np.ndarray
    reference: np.ndarray
    weight_scale: float
    noise_sigma: float
    holdout: np.ndarray

    @property
    def key(self):
        return f"{self.aperture}_{self.frequency_mhz:g}MHz"


def estimate_resources(config):
    """Size guards, not a promise about sparse-LU peak memory."""
    geometry = make_geometry(config)
    active = set(sum(scenario_apertures(config).values(), []))
    x, z = make_grid_axes(config)
    xf, zf = make_grid_axes(config, config["experiment"]["forward_refine"])
    roi = config["reconstruction"]
    nx = int(np.ceil(np.diff(roi["x_bounds_mm"])[0] / roi["spacing_mm"])) + 1
    nz = int(np.ceil(np.diff(roi["z_bounds_mm"])[0] / roi["spacing_mm"])) + 1
    variables = 2 * nx * nz
    rows = sum(len(geometry[name]["positions_mm"])**2 *
               len(config["bands"]["high_mhz" if name == "center_hf" else "low_mhz"])
               for name in active)
    pooled_ports = sum(len(geometry[name]["positions_mm"]) for name in active if name != "center_hf")
    return {
        "inverse_grid_shape": [len(z), len(x)], "inverse_grid_nodes": len(x) * len(z),
        "data_grid_shape": [len(zf), len(xf)], "data_grid_nodes": len(xf) * len(zf),
        "model_shape": [nz, nx], "real_unknowns": variables,
        "unique_complex_observations": rows,
        "retained_jacobian_mib": rows * variables * 16 / 2**20,
        "one_pooled_field_mib": max(pooled_ports, config["geometry"]["center_hf_elements"]) * len(xf) * len(zf) * 16 / 2**20,
        "note": "Sparse LU, temporary arrays, inverse copies and plotting require additional RAM. Frequencies run sequentially.",
    }


def _weights(solver, geometry, names):
    pieces, indices, start = [], {}, 0
    for name in names:
        aperture = geometry[name]
        part = solver.source_weights(aperture["positions_mm"], aperture["normals"], aperture["widths_mm"])
        pieces.append(part)
        indices[name] = np.arange(start, start + part.shape[1])
        start += part.shape[1]
    return sparse.hstack(pieces, format="csc"), indices


def generate_observations(config, geometry, solver, data_solver, basis, truth_coefficients,
                          truth_data, log=print):
    """Share HF data and absolute noise levels across the controlled layouts."""
    blocks, diagnostics = [], []
    experiment = config["experiment"]
    mode = experiment["data_mode"]
    same_grid = experiment["forward_refine"] == 1
    rng = np.random.default_rng(experiment["seed"])
    active = set(sum(scenario_apertures(config).values(), []))
    for band, names in (("low_mhz", ["center_lf", "flat_lf", "arc_lf"]),
                        ("high_mhz", ["center_hf"])):
        names = [name for name in names if name in active]
        weights, indices = _weights(solver, geometry, names)
        data_weights = weights if same_grid else _weights(data_solver, geometry, names)[0]
        reference_name = "center_hf" if band == "high_mhz" else "center_lf"
        for frequency in config["bands"][band]:
            started = perf_counter()
            log(f"  {frequency:g} MHz: background, {', '.join(names)}", flush=True)
            background, diag = solver.solve(frequency, weights, return_diagnostics=True)
            diagnostics.append(dict(kind="inverse_background", **diag))
            jacobians = {name: solver.born_jacobian(frequency, background, indices[name], indices[name], basis)
                         for name in names}
            if mode == "fullwave":
                if same_grid:
                    data_background = background
                else:
                    data_background, diag = data_solver.solve(frequency, data_weights, return_diagnostics=True)
                    diagnostics.append(dict(kind="data_background", **diag))
                reference = data_solver.sample(data_background, data_weights)
                fields, diag = data_solver.solve(frequency, data_weights, *truth_data, return_diagnostics=True)
                diagnostics.append(dict(kind="data_fullwave", **diag))
                scattered = data_solver.sample(fields, data_weights) - reference
                del fields
                nodal_born, diag = data_solver.born_data(frequency, data_weights, *truth_data,
                                                        background_fields=data_background, return_diagnostics=True)
                diagnostics.append(dict(kind="data_nodal_born", **diag))
                clean = {name: scattered[np.ix_(indices[name], indices[name])].ravel() for name in names}
            else:
                # Deliberately matched basis/discretization control, not an
                # independent physical validation of the inverse model.
                reference = solver.sample(background, weights)
                clean = {name: jacobians[name] @ truth_coefficients for name in names}
                nodal_born = None
            del background
            if mode == "fullwave":
                del data_background
            reference_rms = _rms(clean[reference_name])
            # A zero-scattering phantom still yields a defined zero-noise test.
            weight_floor = max(_rms(reference) * 1e-12, 1e-30)
            noise_sigma = (0.0 if experiment["snr_db"] is None else
                           reference_rms * 10**(-float(experiment["snr_db"]) / 20))
            weight_scale = max(noise_sigma if noise_sigma > 0 else reference_rms, weight_floor)
            # LF comparisons have the same port count: use the same complex
            # standardized noise samples, without enforcing reciprocity of noise.
            standardized = (rng.standard_normal(clean[reference_name].size)
                            + 1j * rng.standard_normal(clean[reference_name].size)) / np.sqrt(2)
            for name in names:
                index = indices[name]
                blocks.append(Observation(
                    name, frequency, jacobians[name], clean[name], clean[name] + noise_sigma * standardized,
                    clean[name].copy() if nodal_born is None else nodal_born[np.ix_(index, index)].ravel(),
                    reference[np.ix_(index, index)], weight_scale, noise_sigma,
                    holdout_mask(len(index), experiment["holdout_stride"])))
            log(f"    done in {perf_counter() - started:.2f} s; common noise sigma={noise_sigma:.3g}", flush=True)
    return blocks, diagnostics


def _system(blocks, training=True):
    mask = [(~b.holdout if training else b.holdout) for b in blocks]
    return (np.vstack([b.jacobian[m] / b.weight_scale for b, m in zip(blocks, mask)]),
            np.concatenate([b.observed[m] / b.weight_scale for b, m in zip(blocks, mask)]))


def response_metrics(blocks, predictions):
    answer = {}
    for subset in ("train", "holdout", "all"):
        masks = [(~b.holdout if subset == "train" else b.holdout if subset == "holdout"
                  else np.ones_like(b.holdout)) for b in blocks]
        p = np.concatenate([v[m] for v, m in zip(predictions, masks)])
        observed = np.concatenate([b.observed[m] for b, m in zip(blocks, masks)])
        clean = np.concatenate([b.clean[m] for b, m in zip(blocks, masks)])
        scale = np.concatenate([np.full(np.sum(m), b.weight_scale) for b, m in zip(blocks, masks)])
        answer[subset] = {"complex_samples": int(p.size),
                          "observed_relative_l2_percent": _relative_error(p, observed),
                          "clean_relative_l2_percent": _relative_error(p, clean),
                          "weighted_observed_relative_l2_percent": _relative_error(p / scale, observed / scale)}
    return answer


def image_metrics(estimate, truth, eval_z, config):
    med = config["medium"]
    ep = physical_maps(*estimate, med["c0_m_s"], med["rho0_kg_m3"])
    tp = physical_maps(*truth, med["c0_m_s"], med["rho0_kg_m3"])
    depth = np.broadcast_to(eval_z[:, None], truth[0].shape)
    amplitude = np.maximum(np.abs(truth[0]), np.abs(truth[1]))
    support = amplitude > .05 * max(float(np.max(amplitude)), 1e-15)
    regions = {"all": np.ones_like(support), "shallow_le_15mm": depth <= 15,
               "deep_gt_15mm": depth > 15, "target_support": support, "background": ~support}
    answer = {}
    for name, mask in regions.items():
        if not mask.any():
            continue
        values = {"pixels": int(np.sum(mask))}
        for field, prediction, target in zip(("m_kappa", "m_eta"), estimate, truth):
            p, t = prediction[mask], target[mask]
            values[field + "_rmse_percentage_points"] = 100 * _rms(p - t)
            values[field + "_relative_l2_percent"] = _relative_error(p, t)
            values[field + "_correlation"] = (float(np.corrcoef(p, t)[0, 1])
                                                 if np.std(p) > 1e-15 and np.std(t) > 1e-15 else None)
        for key in ("speed_m_s", "density_kg_m3"):
            values[key + "_rmse"] = _rms(ep[key][mask] - tp[key][mask])
        answer[name] = values
    return answer


def replay_fullwave(config, geometry, solver, data_basis, model, blocks, log=print):
    """Re-solve the recovered medium; no truth or test-channel fitting occurs."""
    n = model.size // 2
    material = (data_basis @ model[:n], data_basis @ model[n:])
    predictions, diagnostics = [], []
    for block in blocks:
        weights = _weights(solver, geometry, [block.aperture])[0]
        fields, diag = solver.solve(block.frequency_mhz, weights, *material, return_diagnostics=True)
        predictions.append((solver.sample(fields, weights) - block.reference).ravel())
        diagnostics.append(dict(kind="reconstruction_fullwave_replay", aperture=block.aperture, **diag))
    return predictions, diagnostics


def run_experiment(config, output_path, *, phantom_path=None, replay=False,
                   max_grid_nodes=400000, max_jacobian_mb=768, log=print):
    """Run all comparisons, write auditable arrays, metrics, figures and provenance."""
    started = perf_counter()
    provenance = _provenance(phantom_path)
    validate_config(config)
    if config["experiment"]["data_mode"] == "born" and config["experiment"]["forward_refine"] != 1:
        raise ValueError("born is a matched-basis control; forward_refine must be 1. Use fullwave for a mismatch test.")
    resources = estimate_resources(config)
    if max(resources["inverse_grid_nodes"], resources["data_grid_nodes"]) > max_grid_nodes:
        raise ValueError(f"Grid exceeds --max-grid-nodes {max_grid_nodes}; inspect --dry-run and raise this guard explicitly if RAM permits")
    if resources["retained_jacobian_mib"] > max_jacobian_mb:
        raise ValueError(f"Jacobians exceed --max-jacobian-mb {max_jacobian_mb}; reduce ports/basis or raise this guard")
    output = Path(output_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --out directory.")
    geometry = make_geometry(config)
    for aperture in geometry.values():
        holdout_mask(len(aperture["positions_mm"]), config["experiment"]["holdout_stride"])
    x, z = make_grid_axes(config)
    xf, zf = make_grid_axes(config, config["experiment"]["forward_refine"])
    med = config["medium"]
    solver = Solver(Grid(x, z, config["grid"]["pml_mm"]), med["c0_m_s"])
    data_solver = solver if config["experiment"]["forward_refine"] == 1 else Solver(
        Grid(xf, zf, config["grid"]["pml_mm"]), med["c0_m_s"])
    basis, mx, mz = make_basis(config, x, z)
    truth_data = make_phantom(config, xf, zf, phantom_path)
    truth_model = make_phantom(config, mx, mz, phantom_path)
    model_truth = np.r_[truth_model[0].ravel(), truth_model[1].ravel()]
    if config["experiment"]["data_mode"] == "born":
        truth_data = tuple((basis @ field.ravel()).reshape(len(z), len(x)) for field in truth_model)
    properties = physical_maps(*truth_data, med["c0_m_s"], med["rho0_kg_m3"])
    minimum_speed = float(np.min(properties["speed_m_s"]))
    ppw = minimum_speed / (1000 * max(config["bands"]["high_mhz"]) * config["grid"]["spacing_mm"])
    if ppw < 8:
        raise ValueError(f"Only {ppw:.2f} inverse-grid points per wavelength in the true medium; reduce grid.spacing_mm")
    # The evaluation grid is fixed independently of the wave discretization.
    # Changing --grid-refine therefore keeps both image metric samples and the
    # reconstruction coefficients identical across the convergence study.
    roi = config["reconstruction"]
    ex = np.linspace(*roi["x_bounds_mm"], int(np.ceil(np.diff(roi["x_bounds_mm"])[0] / .25)) + 1)
    ez = np.linspace(*roi["z_bounds_mm"], int(np.ceil(np.diff(roi["z_bounds_mm"])[0] / .25)) + 1)
    eval_basis, _, _ = make_basis(config, ex, ez)
    continuous_truth = make_phantom(config, ex, ez, phantom_path)
    projected_truth = tuple((eval_basis @ field.ravel()).reshape(len(ez), len(ex)) for field in truth_model)
    eval_truth = projected_truth if config["experiment"]["data_mode"] == "born" else continuous_truth
    bound_truth = truth_model if config["experiment"]["data_mode"] == "born" else truth_data
    truth_maxima = {"m_kappa": float(np.max(np.abs(bound_truth[0]))),
                    "m_eta": float(np.max(np.abs(bound_truth[1])))}
    truth_outside_bounds = any(value > config["inversion"]["bound"] for value in truth_maxima.values())
    if truth_outside_bounds:
        log("WARNING: true material contrast exceeds inversion.bound; this prior excludes the truth. See summary.json.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    _json_write(output / "resolved_config.json", config)
    log(f"Generating {config['experiment']['data_mode']} observations; {resources['real_unknowns']} real unknowns", flush=True)
    blocks, wave_diagnostics = generate_observations(config, geometry, solver, data_solver, basis,
                                                     model_truth, truth_data, log)
    central = [b for b in blocks if b.aperture in ("center_hf", "center_lf")]
    central_system, _ = _system(central)
    norms = np.linalg.norm(central_system, axis=0)
    common_scale = float(np.median(norms[norms > 0])) if np.any(norms > 0) else 1.0
    del central_system
    truth_properties = physical_maps(*eval_truth, med["c0_m_s"], med["rho0_kg_m3"])
    data_basis = (basis if data_solver is solver else make_basis(config, xf, zf)[0]) if replay else None
    results, summary, csv_rows = {}, {}, []
    for name, apertures in scenario_apertures(config).items():
        log(f"Reconstructing {name} ...", flush=True)
        scenario_start = perf_counter()
        selected = [b for b in blocks if b.aperture in apertures]
        jacobian, observed = _system(selected)
        jacobian /= common_scale
        observed /= common_scale
        coefficients, inverse_diagnostics = solve_inverse(jacobian, observed, mx, mz, config["inversion"])
        sensitivity = conditional_sensitivity(jacobian)
        del jacobian, observed
        n = coefficients.size // 2
        estimate = tuple((eval_basis @ v).reshape(len(ez), len(ex)) for v in (coefficients[:n], coefficients[n:]))
        metrics = {
            "apertures": apertures, "optimizer": inverse_diagnostics,
            "image": image_metrics(estimate, eval_truth, ez, config),
            "born_response": response_metrics(selected, [b.jacobian @ coefficients for b in selected]),
            "truth_nodal_born_response": response_metrics(selected, [b.nodal_born for b in selected]),
            "truth_coarse_basis_born_response": response_metrics(selected, [b.jacobian @ model_truth for b in selected]),
        }
        replay_values = None
        if replay:
            replay_values, diagnostics = replay_fullwave(config, geometry, data_solver, data_basis,
                                                         coefficients, selected, log)
            wave_diagnostics.extend(dict(scenario=name, **d) for d in diagnostics)
            metrics["fullwave_replay_response"] = response_metrics(selected, replay_values)
        metrics["elapsed_seconds"] = perf_counter() - scenario_start
        directory = output / name
        directory.mkdir()
        physical = physical_maps(*estimate, med["c0_m_s"], med["rho0_kg_m3"])
        np.savez_compressed(directory / "reconstruction.npz", x_mm=ex, z_mm=ez,
                            m_kappa=estimate[0], m_eta=estimate[1],
                            truth_m_kappa=eval_truth[0], truth_m_eta=eval_truth[1],
                            model_x_mm=mx, model_z_mm=mz,
                            coefficient_m_kappa=coefficients[:n].reshape(len(mz), len(mx)),
                            coefficient_m_eta=coefficients[n:].reshape(len(mz), len(mx)),
                            truth_speed_m_s=truth_properties["speed_m_s"],
                            truth_density_kg_m3=truth_properties["density_kg_m3"], **physical)
        np.savez_compressed(directory / "sensitivity.npz", model_x_mm=mx, model_z_mm=mz, **sensitivity)
        response_arrays = {}
        for i, block in enumerate(selected):
            key = f"block_{i:02d}"
            response_arrays.update({key + "_frequency_mhz": block.frequency_mhz,
                                    key + "_aperture": block.aperture,
                                    key + "_clean": block.clean, key + "_observed": block.observed,
                                    key + "_born_prediction": block.jacobian @ coefficients,
                                    key + "_nodal_born_truth": block.nodal_born,
                                    key + "_coarse_born_truth": block.jacobian @ model_truth,
                                    key + "_holdout_mask": block.holdout,
                                    key + "_weight_scale": block.weight_scale,
                                    key + "_noise_sigma": block.noise_sigma})
            if replay_values is not None:
                response_arrays[key + "_fullwave_replay"] = replay_values[i]
        np.savez_compressed(directory / "responses.npz", **response_arrays)
        _json_write(directory / "metrics.json", metrics)
        results[name] = {"m_kappa": estimate[0], "m_eta": estimate[1], "sensitivity": sensitivity}
        summary[name] = metrics
        for region, values in metrics["image"].items():
            csv_rows.append(dict(scenario=name, region=region, **values))
        if not inverse_diagnostics["converged"]:
            log(f"  WARNING: optimizer KKT check did not converge ({inverse_diagnostics['relative_projected_gradient_inf']:.3g}); inspect metrics.json", flush=True)
        overall = metrics["image"]["all"]
        log(f"  kappa RMSE={overall['m_kappa_rmse_percentage_points']:.3f} pp, "
            f"eta RMSE={overall['m_eta_rmse_percentage_points']:.3f} pp; {metrics['elapsed_seconds']:.2f} s", flush=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    _json_write(output / "wave_diagnostics.json", wave_diagnostics)
    from .plotting import save_figures
    save_figures(output, config, geometry, mx, mz, ex, ez, *eval_truth, results)
    provenance["source_files_changed_during_run"] = provenance["code_sha256"] != _source_hashes()
    document = {"config": config, "resources": resources, "provenance": provenance,
                "minimum_true_speed_m_s": minimum_speed, "minimum_inverse_grid_ppw": ppw,
                "truth_maximum_absolute_contrasts": truth_maxima, "truth_outside_inversion_bounds": truth_outside_bounds,
                "continuous_truth_projection_error": image_metrics(projected_truth, continuous_truth, ez, config),
                "image_truth": "basis-projected matched truth" if config["experiment"]["data_mode"] == "born" else "continuous or imported truth",
                "common_inverse_scale": common_scale,
                "noise_convention": "Per-frequency central clean-scatter RMS sets common complex noise sigma; shared standardized noise across LF layouts; HF data shared.",
                "metric_convention": "Image metrics on fixed 0.25-mm evaluation grid; contrasts are independent; shallow z<=15 mm; support means max(abs(truth contrasts))>5% of peak.",
                "sensitivity_caveat": "2x2 same-node submatrices conditional on all other pixels; not global identifiability.",
                "scenarios": summary, "elapsed_seconds": perf_counter() - started}
    _json_write(output / "summary.json", document)
    log(f"Finished in {document['elapsed_seconds']:.1f} s. Results: {output.resolve()}", flush=True)
    return document
