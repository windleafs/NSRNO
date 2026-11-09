"""Reproduce the train_000 fixed-speed density boundary-width experiment."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .compare_replay import norm
from .dataset import indexed, read_json, write_json
from .fullwave import build_case, initialize, read_reference, solve
from .refine_probe import CENTER_ANGLE, MODE, save


def main():
    manifest_path = Path("out/v5/manifest.json")
    manifest = read_json(manifest_path)
    sample = "train_000"
    row = indexed(manifest["records"])[sample]
    output = Path("out/v5/refined_probe/train_000")
    generator = initialize()
    case = build_case(generator, row)
    original = case["maps"].copy()
    reference = read_reference(generator, manifest)
    radius = np.hypot(case["x"][None, :] - MODE["x_center_m"],
                      case["z"][:, None] - MODE["z_center_m"])
    with np.load(output / "zero_density.npz", allow_pickle=False) as archived:
        scale = norm(archived["rf"])

    results = []
    for width in (0.00005, 0.0):
        basis = 0.5 * (1 - np.tanh((radius - 0.003) / width)) if width else radius <= 0.003
        basis = basis.astype(np.float32)
        for step in (0.005, 0.01):
            responses = []
            for sign in (-1, 1):
                case["maps"] = {**original, "density": original["density"] * np.exp(sign * step * basis)}
                rf, trefs, _ = solve(generator, case, reference, manifest["acquisition"], CENTER_ANGLE)
                if abs(trefs[0] - row["source_tref_s"][CENTER_ANGLE[0]]) > 1e-12:
                    raise ValueError("Clock mismatch")
                path = output / f"density_width_{width:g}_step_{step:g}_{sign}.npz"
                save(path, rf, {"width_m": width, "step": step, "sign": sign, "sample": sample})
                responses.append(rf.astype(np.float64))
            results.append({"width_m": width, "step": step,
                            "relative_odd_rf": norm(responses[1] - responses[0]) / (2 * scale)})

    write_json(output / "edge_width_metrics.json", {
        "sample": sample, "angle_deg": 0, "radius_m": .003,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(), "records": results,
    })
    print(json.dumps(results))


if __name__ == "__main__":
    main()
