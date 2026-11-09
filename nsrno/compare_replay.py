"""Measure replay errors against the archived signed channel RF."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from .dataset import indexed, load_observation, read_json, write_json


def norm(a):
    a = a.astype(np.float64, copy=False)
    return float(np.sqrt(np.sum(a * a)))


def compare(manifest, digest, replay_dir, sample_id):
    row = indexed(manifest["records"])[sample_id]
    archived = load_observation(row, manifest)
    path = replay_dir / (sample_id + ".npz")
    meta = read_json(path.with_suffix(".json"))
    if (meta["id"] != sample_id or meta["manifest_sha256"] != digest
        or meta["reference_fingerprint"] != manifest["reference_fingerprint"]):
        raise ValueError(f"Replay provenance mismatch for {sample_id}")
    ix = meta["angle_indices"]
    with np.load(path, allow_pickle=False) as replayed:
        original = archived.rf[ix].astype(np.float64)
        replay_rf = replayed["rf"]
        if replay_rf.shape != original.shape:
            raise ValueError(f"Unexpected RF replay shape {replay_rf.shape}")
        residual = replay_rf.astype(np.float64) - original
        angles = manifest["acquisition"]["angles_deg"]
        clock = replayed["trefs"] - np.asarray(row["source_tref_s"])[ix]
        return {
            "id": sample_id, "variant": row["variant"],
            "angles_deg": [angles[i] for i in ix],
            "rf_rms_relative": norm(residual) / norm(original),
            "per_angle_rf_rms_relative": [norm(diff) / norm(data)
                                           for diff, data in zip(residual, original)],
            "clock_max_abs_s": float(np.max(np.abs(clock))),
            "c_max_abs_m_s": float(np.max(np.abs(replayed["c"] - archived.c))),
            "m_max_abs": float(np.max(np.abs(replayed["m"] - archived.m))),
            "replay_solve_s": float(sum(meta["seconds_per_angle"])),
        }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    digest = sha256(args.manifest.read_bytes()).hexdigest()
    rows = [compare(manifest, digest, args.replay, sid) for sid in args.ids]
    write_json(args.out, {"manifest_sha256": digest, "records": rows})
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
