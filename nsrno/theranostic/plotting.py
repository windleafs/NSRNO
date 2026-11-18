"""Static figures for the idealized two-dimensional aperture experiment.

Figures preserve physical x/z coordinates, show depth increasing downwards,
and share color limits across acquisitions. Sensitivity is a conditional
two-parameter diagnostic at each coefficient node, not a spatially
marginalized uncertainty estimate.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


_SCENARIOS = (
    ("central_hf", "Central HF"),
    ("central_dual", "Central LF + HF\n(virtual LF reference)"),
    ("flat_dual", "Flat LF + central HF"),
    ("arc_dual", "Arc LF + central HF"),
)


def _axis(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be a finite increasing one-dimensional axis")
    return values


def _field(values, shape, name):
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return values


def _save(fig, output, name):
    try:
        fig.savefig(output / name, dpi=170, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def _draw_aperture(ax, aperture, color, marker, *, virtual=False):
    positions = np.asarray(aperture["positions_mm"], dtype=float)
    normals = np.asarray(aperture["normals"], dtype=float)
    widths = np.asarray(aperture["widths_mm"], dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    if virtual:
        ax.scatter(positions[:, 0], positions[:, 1], s=43, marker=marker,
                   facecolors="none", edgecolors=color, linewidths=1.1, zorder=5)
        return
    segments = np.stack((positions - .5 * widths[:, None] * tangent,
                         positions + .5 * widths[:, None] * tangent), axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=2.2, zorder=4))
    ax.scatter(positions[:, 0], positions[:, 1], s=13, marker=marker,
               color=color, zorder=4)
    ax.quiver(positions[:, 0], positions[:, 1], 2.3 * normals[:, 0],
              2.3 * normals[:, 1], angles="xy", scale_units="xy", scale=1,
              width=.003, color=color, alpha=.8, zorder=3)


def _geometry_figure(output, config, geometry):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.20, top=.84, wspace=.12)
    blue, orange, purple = "#2465a8", "#c76818", "#754299"
    roi, grid = config["reconstruction"], config["grid"]
    x0, x1 = roi["x_bounds_mm"]
    z0, z1 = roi["z_bounds_mm"]
    for ax, name, title in zip(axes, ("flat_lf", "arc_lf"),
                               ("Flat LF side arrays", "Curved LF side arrays")):
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                               facecolor="#eaf0f5", edgecolor="#778899",
                               linewidth=1.0, linestyle="--", zorder=0))
        _draw_aperture(ax, geometry["center_hf"], blue, "s")
        _draw_aperture(ax, geometry[name], orange, "o")
        _draw_aperture(ax, geometry["center_lf"], purple, "D", virtual=True)
        focus = float(config["geometry"]["focus_z_mm"])
        ax.scatter([0], [focus], s=65, marker="+", color="#555555", linewidths=1.3, zorder=2)
        ax.text(x0 + .7, z0 + 1.7, "Unknown ROI", color="#536273", fontsize=9, va="top")
        if z0 < 15 < z1:
            ax.plot([x0, x1], [15, 15], color="#778899", linewidth=.8, linestyle=":")
        ax.set(xlim=grid["x_bounds_mm"], ylim=grid["z_bounds_mm"][::-1],
               xlabel="Lateral x (mm)", title=title, aspect="equal")
        ax.grid(color="#dddddd", alpha=.55, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Depth z (mm)")
    fig.suptitle("Idealized in-plane acquisition geometry", fontsize=15, y=.98)
    fig.text(.5, .91, "Arrows show element normals; line segments show active widths. PML is outside this view.",
             ha="center", fontsize=10, color="#4c5661")
    handles = [
        Line2D([], [], color=blue, marker="s", linestyle="none", markersize=5, label="Central HF"),
        Line2D([], [], color=orange, marker="o", linestyle="none", markersize=5, label="LF side elements"),
        Line2D([], [], color=purple, marker="D", markerfacecolor="none", linestyle="none",
               markersize=6, label="Virtual central LF reference"),
        Patch(facecolor="#eaf0f5", edgecolor="#778899", linestyle="--", label="Reconstruction ROI"),
        Line2D([], [], color="#555555", marker="+", linestyle="none", markersize=7, label="Arc orientation focus"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=3, frameon=False, fontsize=9, columnspacing=2.0)
    _save(fig, output, "geometry.png")


def _map(ax, x, z, values, *, cmap, vmin, vmax):
    artist = ax.pcolormesh(x, z, values, shading="nearest", cmap=cmap,
                           vmin=vmin, vmax=vmax, rasterized=True)
    ax.set(xlim=(x[0], x[-1]), ylim=(z[-1], z[0]), aspect="equal")
    if z[0] < 15 < z[-1]:
        ax.axhline(15, color="#303030", linewidth=.7, linestyle="--", alpha=.65)
    ax.tick_params(labelsize=8)
    return artist


def _reconstruction_figure(output, x, z, truth_kappa, truth_eta, results):
    fig, axes = plt.subplots(2, 5, figsize=(15.2, 9.1), sharex=True, sharey=True,
                              constrained_layout=True)
    titles = ["Truth"] + [title for _, title in _SCENARIOS]
    for row, (name, truth, symbol) in enumerate((
        ("m_kappa", truth_kappa, r"$m_\kappa$ (%)"),
        ("m_eta", truth_eta, r"$m_\eta$ (%)"),
    )):
        values = [truth] + [results[key][name] for key, _ in _SCENARIOS]
        limit = max(max(float(np.max(np.abs(value))) for value in values) * 100, 1e-8)
        for column, value in enumerate(values):
            artist = _map(axes[row, column], x, z, 100 * value,
                          cmap="RdBu_r", vmin=-limit, vmax=limit)
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=10, pad=9)
            else:
                axes[row, column].set_xlabel("Lateral x (mm)", fontsize=9)
        axes[row, 0].set_ylabel("Depth z (mm)", fontsize=9)
        colorbar = fig.colorbar(artist, ax=axes[row, :], fraction=.022, pad=.015, shrink=.93)
        colorbar.set_label(symbol, fontsize=11)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Independent compressibility and inverse-density contrasts\n"
                 "Shared color limits within each row; dashed line marks z = 15 mm", fontsize=14)
    _save(fig, output, "reconstruction.png")


def _sensitivity_figure(output, x, z, results):
    shape = (len(z), len(x))
    ratios, traces = [], []
    for key, _ in _SCENARIOS:
        sensitivity = results[key]["sensitivity"]
        ratio = np.asarray(sensitivity["conditional_eigenvalue_ratio"], dtype=float)
        # The conditional_* names are produced by inverse.conditional_sensitivity.
        trace_key = ("conditional_information_trace" if "conditional_information_trace" in sensitivity
                     else "information_trace")
        trace = np.asarray(sensitivity[trace_key], dtype=float)
        if ratio.size != np.prod(shape) or trace.size != np.prod(shape):
            raise ValueError(f"{key} sensitivity must have one value per model coefficient node")
        ratio = _field(ratio.reshape(shape), shape, f"{key} sensitivity ratio")
        trace = _field(trace.reshape(shape), shape, f"{key} information trace")
        if np.any(ratio < -1e-10) or np.any(ratio > 1 + 1e-10) or np.any(trace < 0):
            raise ValueError("Sensitivity ratios must lie in [0,1] and information traces must be nonnegative")
        ratios.append(np.clip(ratio, 0, 1))
        traces.append(trace)
    positive = np.concatenate([value[value > 0] for value in traces])
    if positive.size:
        low, high = float(np.log10(positive.min())), float(np.log10(positive.max()))
        if high - low < 1e-10:
            low, high = low - .5, high + .5
    else:
        low, high = -1., 0.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d7d7d7")
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 9.1), sharex=True, sharey=True,
                              constrained_layout=True)
    for column, (_, title) in enumerate(_SCENARIOS):
        ratio_artist = _map(axes[0, column], x, z, ratios[column], cmap="viridis", vmin=0, vmax=1)
        log_trace = np.ma.masked_where(traces[column] <= 0,
                                      np.log10(np.maximum(traces[column], np.finfo(float).tiny)))
        trace_artist = _map(axes[1, column], x, z, log_trace, cmap=cmap, vmin=low, vmax=high)
        axes[0, column].set_title(title, fontsize=10, pad=9)
        axes[1, column].set_xlabel("Lateral x (mm)", fontsize=9)
    axes[0, 0].set_ylabel("Depth z (mm)", fontsize=9)
    axes[1, 0].set_ylabel("Depth z (mm)", fontsize=9)
    for row, artist, label in ((0, ratio_artist, r"$\lambda_{\min}/\lambda_{\max}$"),
                               (1, trace_artist, r"$\log_{10}\,\mathrm{tr}(G_j)$")):
        colorbar = fig.colorbar(artist, ax=axes[row, :], fraction=.025, pad=.018, shrink=.93)
        colorbar.set_label(label, fontsize=11)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Conditional two-parameter sensitivity\n"
                 r"Local $G_j=\mathrm{Re}(J_j^H J_j)$ blocks treat all other coefficients as known"
                 "\nShared color limits; zero trace is gray; dashed line marks z = 15 mm", fontsize=12)
    _save(fig, output, "sensitivity.png")


def save_figures(output_path, cfg, geometry, model_x, model_z, eval_x, eval_z,
                 truth_kappa, truth_eta, results):
    """Write geometry, reconstruction, and conditional-sensitivity PNGs.

    Reconstructed and truth contrast maps have shape ``[len(eval_z), len(eval_x)]``.
    Sensitivity vectors use C-order ``[model_z, model_x]`` coefficient ordering.
    Colors show relative material contrasts in percent and retain every finite
    estimated value; no acquisition receives a separately rescaled color axis.
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    model_x, model_z = _axis(model_x, "model_x"), _axis(model_z, "model_z")
    eval_x, eval_z = _axis(eval_x, "eval_x"), _axis(eval_z, "eval_z")
    shape = (len(eval_z), len(eval_x))
    truth_kappa = _field(truth_kappa, shape, "truth_kappa")
    truth_eta = _field(truth_eta, shape, "truth_eta")
    checked_results = {}
    for key, _ in _SCENARIOS:
        checked_results[key] = dict(results[key])
        for material in ("m_kappa", "m_eta"):
            checked_results[key][material] = _field(results[key][material], shape, f"{key}.{material}")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"}):
        _geometry_figure(output, cfg, geometry)
        _reconstruction_figure(output, eval_x, eval_z, truth_kappa, truth_eta, checked_results)
        _sensitivity_figure(output, model_x, model_z, checked_results)
