"""Prepare independent native c/rho and signed RF targets from V5 provenance."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import numpy as np

from .dataset import load_observation, read_json, write_json
from .fullwave import build_case, load_generator
from .rf_representation import IQ_SAMPLES, RF_FS, RF_SAMPLES, encode

TRAIN_CASE = "Neg_07_Left"
VALIDATION_CASE = "Neg_35_Left"
VERSION = 1
GENERATOR = None
MANIFEST = None
OUTPUT = None
DIGEST = None


def initialize(manifest, output, digest):
    global GENERATOR, MANIFEST, OUTPUT, DIGEST
    GENERATOR = load_generator()
    MANIFEST, OUTPUT, DIGEST = manifest, output, digest


def save_array(path, array):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, array, allow_pickle=False)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(row):
    sid = row["id"]
    media = OUTPUT / f"{sid}.medium.npy"
    iq = OUTPUT / f"{sid}.iq.npy"
    receipt = OUTPUT / f"{sid}.json"
    expected = {"id": sid, "version": VERSION, "source_sha256": DIGEST,
                "role": "train" if row["case"] == TRAIN_CASE else "validation",
                "case": row["case"], "variant": row["variant"],
                "angles_deg": MANIFEST["acquisition"]["angles_deg"],
                "source_tref_s": row["source_tref_s"]}
    if receipt.is_file() and media.is_file() and iq.is_file():
        old = read_json(receipt)
        if any(old.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Prepared sample provenance differs: {sid}")
        return old

    case = build_case(GENERATOR, row)
    original = load_observation(row, MANIFEST)
    c, m, _ = GENERATOR.truth_maps(case["maps"], case["x"], case["face"])
    c_error = float(np.max(np.abs(original.c - c)))
    m_error = float(np.max(np.abs(original.m - m)))
    if c_error > .001 or m_error > 1e-5:
        raise ValueError(f"Archive/medium mismatch {sid}: c={c_error:g}, m={m_error:g}")
    target, representation_error = encode(original.rf)
    medium = np.stack([case["maps"][key] for key in ("sound_speed", "density")]).astype(np.float32)
    if not np.isfinite(medium).all() or not (medium > 0).all():
        raise ValueError(f"Invalid physical medium for {sid}")
    save_array(media, medium)
    save_array(iq, target)
    record = {
        **expected, "medium": media.name, "iq": iq.name,
        "medium_shape": list(medium.shape), "iq_shape": list(target.shape),
        "x0_m": float(case["x"][0]), "z0_m": float(case["z"][0]), "dx_m": case["dx"],
        "rf_representation_relative_rms": representation_error,
        "archive_c_max_abs_m_s": c_error, "archive_m_max_abs": m_error,
    }
    write_json(receipt, record)
    return record


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    acquisition = manifest["acquisition"]
    if acquisition["rf_samples"] != RF_SAMPLES or acquisition["fs_hz"] != RF_FS:
        raise ValueError("The RF representation requires the original V5 clock")
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in manifest["records"]
                  if r["case"] == TRAIN_CASE
                  or (r["case"] == VALIDATION_CASE and r["rf_storage"] == "npz")]
    if not any(r["case"] == TRAIN_CASE for r in candidates) or not any(
        r["case"] == VALIDATION_CASE for r in candidates
    ):
        raise ValueError("Missing independent training or validation case")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(manifest, args.out, digest)) as pool:
        for index, record in enumerate(pool.map(prepare, candidates, chunksize=1), start=1):
            records.append(record)
            if index % 25 == 0 or index == len(candidates):
                print(f"Prepared {index}/{len(candidates)} records", flush=True)
    grid_fields = ("medium_shape", "iq_shape", "x0_m", "z0_m", "dx_m")
    grid = {key: records[0][key] for key in grid_fields}
    if any({key: r[key] for key in grid_fields} != grid for r in records):
        raise ValueError("Prepared records do not share a physical grid")
    if grid["iq_shape"] != [len(acquisition["angles_deg"]), 2, acquisition["elements"], IQ_SAMPLES]:
        raise ValueError("Invalid baseband target shape")
    index = {"version": VERSION, "source_sha256": digest, "train_case": TRAIN_CASE,
             "validation_case": VALIDATION_CASE, "acquisition": acquisition, "grid": grid,
             "records": records}
    write_json(args.out / "index.json", index)
    print(json.dumps({"train": sum(r["role"] == "train" for r in records),
                      "validation": sum(r["role"] == "validation" for r in records),
                      "max_rf_representation_relative_rms": max(
                          r["rf_representation_relative_rms"] for r in records)}), flush=True)


if __name__ == "__main__":
    main()
