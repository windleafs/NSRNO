"""CPU-only post-hoc spatial-scale diagnosis; never imported by inversion.

This reads existing material snapshots and evaluation truth. It performs no
propagation, derivative probe, optimization, or checkpoint selection.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from scipy.fft import dctn
from nsrno.v5_materials import evaluation_material


BANDS = [("lambda_ge_4_mm", 0., .25),
         ("lambda_1_to_4_mm", .25, 1.),
         ("lambda_0p2_to_1_mm", 1., 5.),
         ("lambda_lt_0p2_mm", 5., float("inf"))]


def fields(material):
    c, rho, alpha = np.asarray(material[:, 80:921, 178:943], dtype=np.float64)
    return {"c_m_s": c, "density_kg_m3": rho,
            "Z_MRayl": c * rho / 1e6, "alpha_dB_cm_MHz1p1": alpha}


def spectrum(value):
    result = dctn(value - value.mean(), type=2, norm="ortho", workers=1)
    result[0, 0] = 0.
    return result


def main():
    p = ArgumentParser()
    p.add_argument("--sample", default="train_200")
    p.add_argument("--snapshot", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Preserve previous diagnosis")
    truth = fields(evaluation_material(a.sample))
    h, w = next(iter(truth.values())).shape
    # DCT-II is the orthonormal cosine basis of a Neumann rectangle. Wavenumber
    # here is cycles/mm, not radians/mm. All native modes remain in the report.
    kz = np.arange(h, dtype=np.float64) / (2 * h * .05)
    kx = np.arange(w, dtype=np.float64) / (2 * w * .05)
    k = np.hypot(kz[:, None], kx[None, :])
    masks = {name: (k > lo) & (k <= hi) for name, lo, hi in BANDS}
    exact = {name: spectrum(v) for name, v in truth.items()}
    result = {"sample": a.sample, "evaluation_only": True,
              "propagation_or_derivative_evaluations": 0,
              "selection_by_truth": False,
              "domain": {"x_mm": [-19.1, 19.1], "z_mm": [1., 43.],
                         "spacing_mm": .05, "shape": [h, w]},
              "basis": "orthonormal DCT-II; constant removed; Neumann rectangle",
              "interpretation": "spatial scale diagnosis, not an identifiability test",
              "snapshots": []}
    for path in a.snapshot:
        estimate = fields(np.load(path, mmap_mode="r"))
        item = {"material": str(path.resolve()), "fields": {}}
        for name, target in truth.items():
            v = estimate[name]
            coefficients = spectrum(v)
            record = {"mean": float(v.mean()), "truth_mean": float(target.mean()),
                      "std": float(v.std()), "truth_std": float(target.std()),
                      "rmse": float(np.sqrt(np.mean((v - target) ** 2))), "bands": {}}
            for band, mask in masks.items():
                x, y = coefficients[mask], exact[name][mask]
                nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
                record["bands"][band] = {
                    "std": nx / np.sqrt(h * w), "truth_std": ny / np.sqrt(h * w),
                    "amplitude_ratio": nx / ny if ny else None,
                    "correlation": float(np.dot(x, y) / (nx * ny)) if nx * ny else None}
            item["fields"][name] = record
        result["snapshots"].append(item)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["snapshots"]:
        print(json.dumps({"snapshot": row["material"], "fields": row["fields"]}))


if __name__ == "__main__":
    main()
