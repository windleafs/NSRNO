"""Diagnose step-size and spatial-scale effects in the initial physical probe."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .compare_replay import norm
from .dataset import indexed, read_json, write_json
from .fullwave import build_case, initialize, read_reference, solve
from .local_probe import MODE, save, window

GROUPS = {
    "speed": ("c", "smooth", (.00025, .0005)),
    "smooth_density": ("rho", "smooth", (.02, .08)),
    "edge_density": ("rho", "edge", (.005, .01)),
}
BRANCHES = {"speed": ("speed", "smooth_density"), "density": ("edge_density",)}
CENTER_ANGLE = (5,)


def modes(case):
    x = case["x"] - MODE["x_center_m"]
    z = case["z"] - MODE["z_center_m"]
    radius = np.sqrt(x[None, :] ** 2 + z[:, None] ** 2)
    edge = (.5 * (1. - np.tanh((radius - .003) / .0002))).astype(np.float32)
    return {"smooth": window(case), "edge": edge}


def sample(args, manifest, digest):
    args.out.mkdir(parents=True, exist_ok=True)
    generator = initialize()
    row = indexed(manifest["records"])[args.id]
    if row["case"] not in manifest["development_cases"]:
        raise ValueError("Refinement is restricted to development cases")
    case = build_case(generator, row)
    original = case["maps"].copy()
    mode = modes(case)
    reference = read_reference(generator, manifest)

    def run(name, c, rho, parameters):
        metadata = {"sample": args.id, "manifest_sha256": digest, **parameters}
        path = args.out / (name + ".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(saved["metadata_json"].item()) != metadata:
                    raise ValueError(f"Unexpected provenance for {path}")
            return
        case["maps"] = {
            **original, "sound_speed": original["sound_speed"] * np.exp(c),
            "density": original["density"] * np.exp(rho),
        }
        rf, trefs, _ = solve(generator, case, reference, manifest["acquisition"], CENTER_ANGLE)
        if not np.allclose(trefs, np.asarray(row["source_tref_s"])[list(CENTER_ANGLE)],
                           rtol=0, atol=1e-12):
            raise ValueError("Unexpected clock shift")
        save(path, rf, metadata)
        print(f"Saved {path}", flush=True)

    run("zero_" + args.branch, 0., 0., {"branch": args.branch})
    for name in BRANCHES[args.branch]:
        parameter, shape, steps = GROUPS[name]
        for index, step in enumerate(steps):
            for sign in (-1, 1):
                change = sign * step * mode[shape]
                run(f"{name}_{index}_{sign}", change if parameter == "c" else 0.,
                    change if parameter == "rho" else 0.,
                    {"group": name, "step": step, "sign": sign, "mode": shape})
    if args.branch == "density":
        run("mixed_refined", .0004 * mode["smooth"], -.008 * mode["edge"],
            {"mixed_log_c_log_rho": [.0004, -.008]})


def analyze(args, manifest, digest):
    from .dataset import load_observation

    row = indexed(manifest["records"])[args.id]
    archived = load_observation(row, manifest).rf[list(CENTER_ANGLE)].astype(np.float64).ravel()

    def read(name):
        with np.load(args.out / (name + ".npz"), allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].item())
            if metadata["sample"] != args.id or metadata["manifest_sha256"] != digest:
                raise ValueError(f"Mismatched simulation {name}")
            return saved["rf"].astype(np.float64).ravel()

    zeros = [read("zero_" + name) for name in BRANCHES]
    base = .5 * (zeros[0] + zeros[1])
    archive_norm = norm(archived)
    difference = norm(zeros[0] - zeros[1]) / archive_norm
    result = {"sample": args.id, "manifest_sha256": digest, "angle_deg": 0.,
              "zero_repeat_relative": difference,
              "zero_archive_relative": [norm(z - archived) / archive_norm for z in zeros],
              "groups": {}}
    tangents = {}
    for group, (_, _, steps) in GROUPS.items():
        response = [read(f"{group}_{i}_1") - read(f"{group}_{i}_-1") for i in range(len(steps))]
        first, second = [r / (2 * h) for r, h in zip(response, steps)]
        tangents[group] = first
        result["groups"][group] = {
            "steps": steps, "odd_response_relative": [norm(d) / (2 * archive_norm) for d in response],
            "derivative_change_relative": norm(first - second) / norm(first),
            "derivative_cosine": float(np.dot(first, second)) / (norm(first) * norm(second)),
        }

    columns = np.stack((tangents["speed"], tangents["edge_density"]), axis=1)
    delta = read("mixed_refined") - base
    result["mixed_truth"] = [.0004, -.008]
    result["condition_number"] = float(np.linalg.cond(columns))
    result["fits"] = {}
    for name, indices in (("joint", [0, 1]), ("c_only", [0]), ("rho_only", [1])):
        design = columns[:, indices]
        estimate = np.linalg.lstsq(design, delta, rcond=None)[0]
        params = np.zeros(2)
        params[indices] = estimate
        result["fits"][name] = {
            "estimate": params.tolist(),
            "residual_rms_relative": norm(design @ estimate - delta) / norm(delta),
        }
    write_json(args.out / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("simulate", "analyze"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--id", default="train_000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--branch", choices=tuple(BRANCHES))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    if args.command == "simulate":
        if args.branch is None:
            parser.error("--branch is required for simulation")
        sample(args, manifest, digest)
    else:
        analyze(args, manifest, digest)


if __name__ == "__main__":
    main()
