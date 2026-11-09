"""Verify RF shapes, units and sample identity through the independent loader."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import numpy as np

from .dataset import load_observation, read_json, write_json


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    counts = Counter()
    failures = []
    extrema = {}
    for index, row in enumerate(manifest["records"], start=1):
        try:
            observation = load_observation(row, manifest)
            amplitude = float(np.sqrt(np.mean(np.square(observation.rf, dtype=np.float64))))
            if not amplitude > 0:
                raise ValueError("Zero RF energy")
            kind = row["variant"]
            counts[kind] += 1
            low, high = extrema.get(kind, (amplitude, amplitude))
            extrema[kind] = [min(low, amplitude), max(high, amplitude)]
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            failures.append({"id": row["id"], "error": str(error)})
        if index % 100 == 0:
            print(f"Verified {index} rows, {len(failures)} failures", flush=True)
    write_json(args.out, {"manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
                          "count": sum(counts.values()), "counts_by_variant": counts,
                          "rf_rms_range_by_variant": extrema, "failures": failures})
    if failures:
        raise SystemExit(f"{len(failures)} RF records failed validation. See {args.out}")
    print(f"Verified all {sum(counts.values())} records", flush=True)


if __name__ == "__main__":
    main()
