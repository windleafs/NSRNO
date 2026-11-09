"""Deterministic recipes for the archived V5 supplemental acquisitions.

Recipe provenance: the original V5 gen_decoupled_raw.py. Parameters, array
precision and random draw order are part of the archived dataset definition.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def edit_density(rho, x, z, seed):
    zz, xx = np.meshgrid(z * 1000, x * 1000, indexing="ij")
    rng = np.random.default_rng(seed)
    result = rho.copy()
    macro = uniform_filter(rho, size=int(1e-3 / 5e-5) | 1)
    while True:
        cx, cz = rng.uniform(-13., 13.), rng.uniform(10., 38.)
        radius = rng.uniform(2.5, 4.)
        mask = (xx - cx)**2 + (zz - cz)**2 <= radius**2
        if mask.mean() >= 1e-4:
            result[mask] = macro[mask]
            break

    for hyper in (True, False):
        while True:
            width, height = rng.uniform(6., 12.), rng.uniform(6., 12.)
            x0 = rng.uniform(-16., 16. - width)
            z0 = rng.uniform(8., 38. - height)
            mask = ((xx >= x0) & (xx <= x0 + width)
                    & (zz >= z0) & (zz <= z0 + height))
            if mask.mean() >= 1e-4:
                break
        if hyper:
            result[mask] += rng.normal(0., 12., result.shape)[mask]
        else:
            result[mask] = macro[mask] + .35 * (result[mask] - macro[mask])
    return np.clip(result, 800., 1300.).astype(np.float32)


def make_variant(generator, plane, case, seed, variant):
    if variant == "calibuni":
        shape = (len(case["z"]), len(case["x"]))
        rng = np.random.default_rng(seed + 606)
        mult = rng.uniform(.3, 2.2)
        speckle = rng.normal(0., 8. * mult, shape) * (case["z"][:, None] * 1e3 >= 0.)
        speed = rng.uniform(1450., 1580.)
        return {
            "sound_speed": np.full(shape, speed, np.float32),
            "density": (1000. + speckle).astype(np.float32),
            "alpha_coeff": np.full(shape, .002, np.float32),
            "BonA": np.zeros(shape, np.float32),
        }

    offset = {"reseed": 101, "texture": 101, "cswap": 0, "calibhom": 202}[variant]
    maps, codes, *_ = generator.medium_builder.build_medium(
        plane, case["x"], case["z"], seed=seed + offset, preset="dual_scale")
    if variant == "texture":
        maps["density"] = edit_density(maps["density"], case["x"], case["z"], seed + 303)
    elif variant == "cswap":
        rng = np.random.default_rng(seed + 404)
        gland = rng.uniform(1510., 1590.)
        fat = rng.uniform(1445., 1500.)
        sigma = int(0.2e-3 / 5e-5) | 1
        s_g = gaussian_filter((codes == 2).astype(np.float32), sigma)
        s_f = gaussian_filter((codes == 3).astype(np.float32), sigma)
        delta = s_g * (gland - 1560.) + s_f * (fat - 1455.)
        maps["sound_speed"] = np.clip(maps["sound_speed"] + delta, 1400., 1700.).astype(np.float32)
    elif variant == "calibhom":
        speed = np.random.default_rng(seed + 505).uniform(1450., 1580.)
        maps["sound_speed"] = np.full_like(maps["sound_speed"], speed)
    return maps
