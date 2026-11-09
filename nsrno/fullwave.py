"""Replay original UltraWave channel RF, in a process without PyTorch."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

import h5py
import numpy as np

from .dataset import indexed, read_json, write_json

TOOLCHAIN = Path("/home/zhuangyang/fmmodel/neural_asp")
COMPILER = Path("/data/zhuangyang/nvhpc_2023_237_Linux_x86_64_cuda_12.2/install_components/Linux_x86_64/23.7/compilers/bin")


def load_generator():
    if "torch" in sys.modules:
        raise RuntimeError("UltraWave must run separately from PyTorch")
    sys.path.insert(0, str(TOOLCHAIN))
    from scripts import generate_l11_ultrawave_raw
    return generate_l11_ultrawave_raw


def build_case(generator, row):
    case = generator.geometry_case()
    with h5py.File(row["phantom_h5"], "r") as file:
        plane = np.asarray(file["phan"][row["z_index"]])
    seed = row["scatter_seed"]
    variant = row["variant"]
    if variant in ("oa_breast_original", "high_echo_circles_v1"):
        maps, codes, *_ = generator.medium_builder.build_medium(
            plane, case["x"], case["z"], seed=seed, preset="dual_scale")
        if variant == "high_echo_circles_v1":
            for retry in range(8):
                try:
                    maps, _, _ = generator.add_echogenic_circles(
                        maps, codes, case["x"], case["z"],
                        seed=seed + 404 + 10007 * retry)
                    break
                except RuntimeError:
                    if retry == 7:
                        raise
    else:
        from .variants import make_variant
        maps = make_variant(generator, plane, case, seed, variant)
    case["maps"] = maps
    generator.bench.validate_case(case)
    return case


def read_reference(generator, manifest):
    acq = manifest["acquisition"]
    with np.load(manifest["reference_file"], allow_pickle=False) as archive:
        if (archive["fingerprint"].item() != manifest["reference_fingerprint"]
            or int(archive["nt"]) != acq["native_nt"] or float(archive["dt_s"]) != acq["native_dt_s"]
            or not np.allclose(archive["angles_deg"], acq["angles_deg"], rtol=0, atol=1e-10)):
            raise ValueError("Reference fingerprint or acquisition mismatch")
        reference = archive["rf_native"]
    if reference.shape != (generator.NT, acq["elements"], len(acq["angles_deg"])):
        raise ValueError(f"Unexpected native reference dimensions {reference.shape}")
    return reference


def solve(generator, case, reference, acquisition, angle_indices):
    maps = case["maps"]
    fit = generator.absorption_model(case)
    solver = generator.solver_for(case, maps, fit)
    rf = []
    trefs = []
    times = []
    for index in angle_indices:
        angle = acquisition["angles_deg"][index]
        tref = generator.set_angle(solver, case, angle)
        total, timing = solver.run()
        filtered, _ = generator.sim.analytic_channels(
            total - reference[:, :, index], generator.DT, band=acquisition["band_hz"])
        rf.append(filtered.T.astype(np.float32))
        trefs.append(tref)
        times.append(float(timing["solve_readback_s"]))
        print(json.dumps({"angle_deg": angle, "solve_s": times[-1]}), flush=True)
    result = np.stack(rf)
    if (result.shape != (len(angle_indices), acquisition["elements"], acquisition["rf_samples"])
        or not np.isfinite(result).all()):
        raise ValueError("Invalid full-wave RF")
    return result, np.asarray(trefs), times


def initialize():
    generator = load_generator()
    os.environ["PATH"] = str(COMPILER) + os.pathsep + os.environ["PATH"]
    generator.configure_gpu()
    return generator


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--angle-index", type=int, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / (args.id + ".npz")
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest = read_json(args.manifest)
    row = indexed(manifest["records"])[args.id]
    indices = args.angle_index or list(range(manifest["rf_shape"][0]))
    if len(indices) != len(set(indices)) or not all(0 <= i < manifest["rf_shape"][0] for i in indices):
        parser.error("Angle indices must be unique and within the acquisition")
    generator = initialize()
    case = build_case(generator, row)
    maps = case["maps"]
    c, m, _ = generator.truth_maps(maps, case["x"], case["face"])
    reference = read_reference(generator, manifest)
    rf, trefs, times = solve(generator, case, reference, manifest["acquisition"], indices)
    metadata = {
        "id": row["id"], "variant": row["variant"], "angle_indices": indices,
        "native_shape_zx": maps["sound_speed"].shape, "native_dx_m": case["dx"],
        "medium_sha256": {key: sha256(value.tobytes()).hexdigest()
                          for key, value in maps.items()},
        "generator_sha256": sha256(Path(generator.__file__).read_bytes()).hexdigest(),
        "reference_fingerprint": manifest["reference_fingerprint"],
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "seconds_per_angle": times,
    }
    temp_path = result_path.with_suffix(".partial.npz")
    np.savez(temp_path, rf=rf, c=c, m=m, trefs=trefs)
    temp_path.replace(result_path)
    write_json(result_path.with_suffix(".json"), metadata)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
