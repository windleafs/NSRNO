"""Configuration, hypothetical in-plane arrays, and independent material phantoms.

All public coordinates use millimetres and array order is ``[z, x]``. These
utilities deliberately do not import the archived V5 data loaders. The side
arrays are independently addressed virtual elements, not a model of a named
commercial therapy probe. A common elevation-row probe cannot be represented
by silently projecting its elements into this two-dimensional geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator


def _positive(value, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return result


def _integer(value, name: str, minimum: int = 1) -> int:
    result = _positive(value, name, zero=minimum == 0)
    if result != int(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _bounds(value, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite increasing numeric bounds")
    if any(isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        raise ValueError(f"{name} bounds must be JSON numbers")
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or array[1] <= array[0]:
        raise ValueError(f"{name} must contain two finite increasing bounds")
    return float(array[0]), float(array[1])


def validate_config(config: dict) -> dict:
    """Validate a fully specified JSON configuration, returning it unchanged.

    Positivity and actual points per wavelength are also checked against the
    generated or imported material by the experiment runner. This function
    validates the background sampling, geometry, and user-specified controls.
    """
    required = ("geometry", "bands", "grid", "reconstruction", "medium", "experiment", "inversion")
    if not isinstance(config, dict) or any(name not in config for name in required):
        raise ValueError(f"Configuration requires sections {required}")
    g, bands, grid, roi, medium, experiment, inverse = (config[name] for name in required)
    if any(not isinstance(section, dict) for section in (g, bands, grid, roi, medium, experiment, inverse)):
        raise ValueError("Configuration sections must be JSON objects")
    try:
        if g["plane"] != "xz":
            raise ValueError("Only in-plane xz arrays are supported; elevation rows require a 3-D model")
        for name in ("center_width_mm", "arc_radius_mm", "focus_z_mm"):
            _positive(g[name], f"geometry.{name}")
        for name in ("hf_element_width_mm", "lf_element_width_mm"):
            _positive(g[name], f"geometry.{name}", zero=True)
        _integer(g["center_hf_elements"], "geometry.center_hf_elements", 4)
        _integer(g["side_elements"], "geometry.side_elements", 2)
        amin, amax = (_positive(g[key], f"geometry.{key}") for key in ("side_angle_min_deg", "side_angle_max_deg"))
        if not (np.isfinite([amin, amax]).all() and 0 < amin < amax < 90):
            raise ValueError("Side arc angles must satisfy 0 < min < max < 90 degrees")
        if not isinstance(g["side_receive"], bool):
            raise ValueError("geometry.side_receive must be true or false")
        for band in ("low_mhz", "high_mhz"):
            if not isinstance(bands[band], list) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bands[band]):
                raise ValueError(f"bands.{band} must be a list of JSON numbers")
            f = np.asarray(bands[band], dtype=float)
            if f.ndim != 1 or len(f) == 0 or not np.isfinite(f).all() or np.any(f <= 0) or np.any(np.diff(f) <= 0):
                raise ValueError(f"bands.{band} must be a nonempty, strictly increasing list of positive MHz values")
        if max(bands["low_mhz"]) >= min(bands["high_mhz"]):
            raise ValueError("This benchmark requires disjoint ordered LF/HF bands; it never translates frequencies")
        h = _positive(grid["spacing_mm"], "grid.spacing_mm")
        _positive(grid["pml_mm"], "grid.pml_mm")
        gx, gz = (_bounds(grid[key], f"grid.{key}") for key in ("x_bounds_mm", "z_bounds_mm"))
        rx, rz = (_bounds(roi[key], f"reconstruction.{key}") for key in ("x_bounds_mm", "z_bounds_mm"))
        _positive(roi["spacing_mm"], "reconstruction.spacing_mm")
        if not (gx[0] < rx[0] < rx[1] < gx[1] and gz[0] < rz[0] < rz[1] < gz[1]):
            raise ValueError("The reconstruction ROI must lie strictly inside the physical grid, outside PML")
        if rz[0] <= 0:
            raise ValueError("The reconstruction ROI must start below the central array at z=0")
        c0 = _positive(medium["c0_m_s"], "medium.c0_m_s")
        _positive(medium["rho0_kg_m3"], "medium.rho0_kg_m3")
        _positive(medium["contrast_scale"], "medium.contrast_scale", zero=True)
        background_ppw = c0 / (max(bands["high_mhz"]) * 1000.0 * h)
        if background_ppw < 8:
            raise ValueError(f"Grid has only {background_ppw:.2f} background points per shortest wavelength; require >=8")
        if experiment["data_mode"] not in ("fullwave", "born"):
            raise ValueError("experiment.data_mode must be 'fullwave' or 'born'")
        snr = experiment["snr_db"]
        if snr is not None and (isinstance(snr, bool) or not np.isfinite(float(snr)) or float(snr) <= 0):
            raise ValueError("experiment.snr_db must be positive or null for no noise")
        _integer(experiment["seed"], "experiment.seed", 0)
        _integer(experiment["holdout_stride"], "experiment.holdout_stride", 2)
        _integer(experiment["forward_refine"], "experiment.forward_refine")
        for name in ("lambda0", "lambda_gradient"):
            _positive(inverse[name], f"inversion.{name}", zero=True)
        _integer(inverse["max_iterations"], "inversion.max_iterations")
        bound = _positive(inverse["bound"], "inversion.bound")
        if bound >= 1:
            raise ValueError("inversion.bound must be <1 so 1+m_kappa and 1+m_eta stay positive")
    except KeyError as error:
        raise ValueError(f"Missing required configuration key: {error.args[0]}") from error
    # Check the entire element segment, not just its centre, against the known
    # coupling medium. Arc elements may sit below shallow targets in depth but
    # must remain laterally outside the entire unknown rectangular ROI.
    for name, aperture in make_geometry(config).items():
        p, n, w = (aperture[key] for key in ("positions_mm", "normals", "widths_mm"))
        tangent = np.column_stack((n[:, 1], -n[:, 0]))
        endpoints = np.concatenate((p - .5 * w[:, None] * tangent, p + .5 * w[:, None] * tangent))
        if np.any((endpoints[:, 0] <= gx[0] + h) | (endpoints[:, 0] >= gx[1] - h) |
                  (endpoints[:, 1] <= gz[0] + h) | (endpoints[:, 1] >= gz[1] - h)):
            raise ValueError(f"{name} element segments must lie inside the physical grid with a one-cell margin")
        # A segment's bounding box gives a conservative intersection test with
        # the ROI (it can reject a segment near a corner, but cannot miss one).
        lo = p - .5 * w[:, None] * np.abs(tangent)
        hi = p + .5 * w[:, None] * np.abs(tangent)
        intersects = (hi[:, 0] >= rx[0]) & (lo[:, 0] <= rx[1]) & (hi[:, 1] >= rz[0]) & (lo[:, 1] <= rz[1])
        if np.any(intersects):
            raise ValueError(f"{name} intersects the unknown ROI; arrays must lie in known coupling medium")
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    return validate_config(config)


def make_geometry(config: dict) -> dict[str, dict[str, np.ndarray]]:
    """Return idealized centre, flat-side, and curved-side element layouts.

    Normals point toward the focus and determine the orientation of reciprocal
    finite line elements. They are not an extra ad-hoc cosine directivity law.
    The solver models finite lines in a fluid, without a rigid transducer back
    wall. The virtual LF centre array is a frequency-only comparison, not a
    claim that the HF hardware can receive the LF band.
    """
    g = config["geometry"]
    half = .5 * float(g["center_width_mm"])
    n_side = int(g["side_elements"])
    theta = np.deg2rad(np.linspace(g["side_angle_min_deg"], g["side_angle_max_deg"], n_side))
    theta = np.r_[-theta[::-1], theta]
    radius, focus = float(g["arc_radius_mm"]), float(g["focus_z_mm"])
    arc = np.column_stack((radius * np.sin(theta), focus - radius * np.cos(theta)))
    arc_normal = np.column_stack((-np.sin(theta), np.cos(theta)))

    def aperture(positions, normals, width):
        return {"positions_mm": np.asarray(positions, dtype=float),
                "normals": np.asarray(normals, dtype=float),
                "widths_mm": np.full(len(positions), float(width))}

    def centre(count, width):
        positions = np.column_stack((np.linspace(-half, half, count), np.zeros(count)))
        return aperture(positions, np.tile([0., 1.], (count, 1)), width)

    return {
        "center_hf": centre(int(g["center_hf_elements"]), g["hf_element_width_mm"]),
        "center_lf": centre(2 * n_side, g["lf_element_width_mm"]),
        "flat_lf": aperture(np.column_stack((arc[:, 0], np.zeros(len(arc)))),
                            np.tile([0., 1.], (len(arc), 1)), g["lf_element_width_mm"]),
        "arc_lf": aperture(arc, arc_normal, g["lf_element_width_mm"]),
    }


def make_grid_axes(config: dict, refine: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Uniform solver axes including an absorbing region of at least pml_mm.

    Both the requested interval and PML cell count round outwards. The solver's
    physical bounds therefore extend by less than one cell below the requested
    lower bound and less than two cells above its upper bound. Both axes use
    exactly the same spacing. Refinement divides that spacing; it is independent
    of the reconstruction basis spacing.
    """
    refine = _integer(refine, "refine")
    g = config["grid"]
    h = float(g["spacing_mm"]) / refine
    pad_cells = int(np.ceil(float(g["pml_mm"]) / h))
    axes = []
    for name in ("x_bounds_mm", "z_bounds_mm"):
        lo, hi = _bounds(g[name], name)
        n = int(np.ceil((hi - lo) / h))
        axes.append(lo + h * np.arange(-pad_cells, n + pad_cells + 1))
    return tuple(axes)


def _model_axis(bounds, spacing):
    lo, hi = _bounds(bounds, "model bounds")
    n = int(np.ceil((hi - lo) / float(spacing)))
    return np.linspace(lo, hi, n + 1)


def make_basis(config: dict, x_mm, z_mm):
    """Bilinear interpolation from real coefficient maps to the solver grid.

    Returns ``(basis, model_x_mm, model_z_mm)``. Basis rows and coefficient
    columns use C-order ``[z,x]``. Rows outside the declared ROI are zero. Thus
    values need not continue smoothly across its boundary; the builtin phantom
    has compact support away from that boundary. The support restriction is
    shared by every acquisition and is an explicit prior, not inferred tissue.
    """
    roi = config["reconstruction"]
    mx = _model_axis(roi["x_bounds_mm"], roi["spacing_mm"])
    mz = _model_axis(roi["z_bounds_mm"], roi["spacing_mm"])
    x, z = np.asarray(x_mm, dtype=float), np.asarray(z_mm, dtype=float)
    xx, zz = np.meshgrid(x, z)
    keep = (xx >= mx[0]) & (xx <= mx[-1]) & (zz >= mz[0]) & (zz <= mz[-1])
    rows = np.flatnonzero(keep)
    xp, zp = xx.ravel()[rows], zz.ravel()[rows]
    ix = np.clip(np.searchsorted(mx, xp, side="right") - 1, 0, len(mx) - 2)
    iz = np.clip(np.searchsorted(mz, zp, side="right") - 1, 0, len(mz) - 2)
    tx, tz = (xp - mx[ix]) / (mx[ix + 1] - mx[ix]), (zp - mz[iz]) / (mz[iz + 1] - mz[iz])
    cols = np.stack((iz * len(mx) + ix, iz * len(mx) + ix + 1,
                     (iz + 1) * len(mx) + ix, (iz + 1) * len(mx) + ix + 1))
    values = np.stack(((1 - tz) * (1 - tx), (1 - tz) * tx, tz * (1 - tx), tz * tx))
    basis = sparse.coo_matrix((values.ravel(), (np.tile(rows, 4), cols.ravel())),
                              shape=(len(z) * len(x), len(mz) * len(mx))).tocsr()
    basis.eliminate_zeros()
    return basis, mx, mz


def _compact_ellipse(x, z, x0, z0, rx, rz):
    r2 = ((x - x0) / rx)**2 + ((z - z0) / rz)**2
    return np.maximum(1.0 - r2, 0.0)**2


def make_phantom(config: dict, x_mm, z_mm, material_path: str | Path | None = None):
    """Independent relative compressibility and inverse-density truth maps.

    Custom NPZ files contain ``x_mm``, ``z_mm``, ``speed_m_s``, and
    ``density_kg_m3``, with materials ordered ``[z,x]``. Their axes must cover
    the whole reconstruction ROI. Any supplied material outside that ROI must
    be homogeneous background: it is never silently discarded. The external
    solver region is known coupling fluid. No attenuation map is used.
    """
    xx, zz = np.meshgrid(np.asarray(x_mm, dtype=float), np.asarray(z_mm, dtype=float))
    roi, medium = config["reconstruction"], config["medium"]
    xb, zb = roi["x_bounds_mm"], roi["z_bounds_mm"]
    inside = (xx >= xb[0]) & (xx <= xb[1]) & (zz >= zb[0]) & (zz <= zb[1])
    if material_path is not None:
        with np.load(material_path, allow_pickle=False) as archive:
            required = {"x_mm", "z_mm", "speed_m_s", "density_kg_m3"}
            if not required.issubset(archive.files):
                raise ValueError(f"Material NPZ must include {sorted(required)}")
            x, z, c, rho = (np.asarray(archive[k], dtype=float) for k in
                            ("x_mm", "z_mm", "speed_m_s", "density_kg_m3"))
        for axis, name, bounds in ((x, "x_mm", xb), (z, "z_mm", zb)):
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError(f"Material {name} must be a finite strictly increasing one-dimensional axis")
            if axis[0] > bounds[0] or axis[-1] < bounds[1]:
                raise ValueError(f"Material {name} must cover the complete reconstruction ROI; extrapolation is disabled")
        for field, name in ((c, "speed_m_s"), (rho, "density_kg_m3")):
            if field.shape != (len(z), len(x)) or not np.isfinite(field).all() or np.any(field <= 0):
                raise ValueError(f"Material {name} must be positive finite [z,x] values matching the axes")
        kappa = medium["rho0_kg_m3"] * medium["c0_m_s"]**2 / (rho * c**2) - 1
        eta = medium["rho0_kg_m3"] / rho - 1
        tx, tz = np.meshgrid(x, z)
        outside = (tx < xb[0]) | (tx > xb[1]) | (tz < zb[0]) | (tz > zb[1])
        if np.any(np.abs(kappa[outside]) > 1e-6) or np.any(np.abs(eta[outside]) > 1e-6):
            raise ValueError("Material contrast extends outside the reconstruction ROI; enlarge the ROI instead of dropping unknown tissue")
        query = np.column_stack((zz[inside], xx[inside]))
        outputs = []
        for field in (kappa, eta):
            result = np.zeros_like(xx)
            result[inside] = RegularGridInterpolator((z, x), field, bounds_error=True)(query)
            outputs.append(result)
        return tuple(outputs)

    # Locations are fractions of the ROI, so changing field of view does not
    # accidentally truncate a target. Compact smooth shapes are evaluated on
    # the actual simulation mesh, not projected into the inverse basis.
    xmid, spanx, spanz = .5 * (xb[0] + xb[1]), xb[1] - xb[0], zb[1] - zb[0]
    x = (xx - xmid) / spanx
    z = (zz - zb[0]) / spanz
    shallow_k = _compact_ellipse(x, z, -.21, .18, .13, .075)
    shallow_eta = _compact_ellipse(x, z, .22, .27, .12, .09)
    deep_joint = _compact_ellipse(x, z, -.04, .72, .20, .12)
    deep_eta = _compact_ellipse(x, z, .29, .78, .065, .09)
    ribbon = np.maximum(1 - (x / .40)**2, 0)**2 * np.maximum(
        1 - ((z - .43 - .045 * np.sin(13 * x)) / .035)**2, 0)**2
    scale = float(medium["contrast_scale"])
    kappa = scale * (shallow_k + .55 * ribbon - .85 * deep_joint)
    eta = scale * (-.85 * shallow_eta + .65 * deep_joint + .70 * deep_eta)
    kappa[~inside], eta[~inside] = 0., 0.
    if np.any(kappa <= -1) or np.any(eta <= -1):
        raise ValueError("Phantom scale produces nonpositive compressibility or inverse density")
    return kappa, eta


def physical_maps(m_kappa, m_eta, c0_m_s: float, rho0_kg_m3: float):
    """Convert independent relative contrasts to speed, density, κ and η."""
    kr, er = 1 + np.asarray(m_kappa), 1 + np.asarray(m_eta)
    if not np.isfinite(kr).all() or not np.isfinite(er).all() or np.any(kr <= 0) or np.any(er <= 0):
        raise ValueError("Physical conversion requires positive finite relative parameters")
    return {"speed_m_s": float(c0_m_s) * np.sqrt(er / kr),
            "density_kg_m3": float(rho0_kg_m3) / er,
            "compressibility_pa_inv": kr / (float(rho0_kg_m3) * float(c0_m_s)**2),
            "inverse_density_m3_kg": er / float(rho0_kg_m3)}
