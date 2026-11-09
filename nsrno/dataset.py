"""Read and verify the V5 dataset without importing Geo-Flow."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import numpy as np

DATA = Path("/data/zhuangyang")
BASE_RF = DATA / "NumerialBreastPhantoms/l11_ultrawave_500_11angle"
BASE_CACHE = DATA / "geometry_flow_v2_cache"
SUPPLEMENT = DATA / "geometry_flow_v2_decv4_cache"
SOURCES = (DATA / "tmp/decoupled_root", DATA / "tmp/calib_root")
HOLDOUT = "Neg_47_Left"
DEV_CASES = ("Neg_07_Left", "Neg_35_Left")
RF_AXES = ("angle", "receiver", "time")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(data, temp, indent=2, allow_nan=False)
            temp.write("\n")
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def indexed(rows):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample IDs")
    return by_id


def existing(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def build_manifest(base_cache, supplement, source_roots):
    base_path = base_cache / "manifest.json"
    supplement_path = supplement / "manifest.json"
    base = read_json(base_path)
    extras = read_json(supplement_path)
    raw_root = Path(base["data_root"])
    base_index_path = raw_root / "index.json"
    base_index = read_json(base_index_path)
    parents = indexed(base_index["samples"])
    cache_parents = indexed(base["records"])
    acquisition = base_index["acquisition"]
    shape = (len(acquisition["angles_deg"]), acquisition["elements"],
             acquisition["rf_samples"])
    sources = {}
    input_files = [base_path, supplement_path, base_index_path]
    for root in source_roots:
        path = root / "index.json"
        source_index = read_json(path)
        input_files.append(path)
        source_acq = source_index["acquisition"].copy()
        source_angles = source_acq.pop("angles_deg")
        if (source_acq != {k: v for k, v in acquisition.items() if k != "angles_deg"}
            or not np.allclose(source_angles, acquisition["angles_deg"], rtol=0, atol=1e-10)):
            raise ValueError(f"Acquisition mismatch in {root}")
        for key, info in indexed(source_index["samples"]).items():
            if key in sources:
                raise ValueError(f"Ambiguous supplement source for {key}")
            sources[key] = (root, info)

    rows = []
    for cache, data, is_base in ((base_cache, base, True),
                                 (supplement, extras, False)):
        for rec in data["records"]:
            sid = rec["id"]
            if not is_base:
                source_root, source = sources[sid]
            else:
                source_root, source = raw_root, parents[sid]
            parent_id = sid if is_base else source["source_id"]
            parent = parents[parent_id]
            case = parent["case"]
            if (source["status"] != "complete" or rec["status"] != "complete"
                or parent["split"] != rec["split"] or source["split"] != rec["split"]
                or cache_parents[parent_id]["split"] != rec["split"]
                or (case == HOLDOUT) != (rec["split"] == "test")
                or source["path"] != rec["source_path"]
                or not np.allclose(rec["angles_deg"], acquisition["angles_deg"],
                                   rtol=0, atol=1e-10)
                or rec["fs_hz"] != acquisition["fs_hz"]
                or rec["pitch_m"] != acquisition["pitch_m"]):
                raise ValueError(f"Invalid lineage or acquisition for {sid}")
            shard = existing(source_root / source["path"])
            rf_file = existing(raw_root / source["raw_path"]) if is_base else shard
            variant = parent.get("phantom_variant", "oa_breast_original") if is_base else source["variant"]
            rows.append({
                "id": sid, "parent_id": parent_id, "case": case,
                "anatomy": parent["base_anatomy_id"], "legacy_split": rec["split"],
                "variant": variant, "phantom_h5": existing(Path(parent["h5"])),
                "z_index": parent["z_index"], "scatter_seed": parent["scatter_seed"],
                "v5_cache": existing(cache / rec["cache_file"]),
                "v5_target_resampling": rec["target_resampling"],
                "rf_file": rf_file, "rf_storage": "npz" if is_base else "pt",
                "source_tref_s": rec["tx_t_ref_s"],
            })
    indexed(rows)
    for row in rows:
        if row["case"] not in (*DEV_CASES, HOLDOUT):
            raise ValueError(f"Unexpected case {row['case']}")
        if len(row["source_tref_s"]) != shape[0]:
            raise ValueError(f"Unexpected transmit clock count for {row['id']}")
    return {
        "version": 1, "acquisition": acquisition, "rf_shape": shape, "rf_axes": RF_AXES,
        "coarse_source_grid_zx": base_index["model_grid"], "v5_grid_xz": base["grid"],
        "reference_file": existing(raw_root / "reference_native.npz"),
        "reference_fingerprint": base_index["reference_fingerprint"],
        "development_cases": DEV_CASES, "test_case": HOLDOUT,
        "source_index_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                                for path in input_files},
        "records": rows,
    }


@dataclass(frozen=True)
class Observation:
    rf: np.ndarray
    c: np.ndarray
    m: np.ndarray


def load_observation(row, manifest):
    path = Path(row["rf_file"])
    if row["rf_storage"] == "npz":
        with np.load(path, allow_pickle=False) as saved:
            item = Observation(saved["rf"], saved["c"], saved["m"])
            meta = json.loads(saved["metadata_json"].item())
    elif row["rf_storage"] == "pt":
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=True)
        item = Observation(*(saved[key].numpy() for key in ("rf", "c", "m")))
        meta = saved["metadata"]
    else:
        raise ValueError(f"Unknown RF storage for {row['id']}")

    acq = manifest["acquisition"]
    grid = manifest["coarse_source_grid_zx"]
    label_shape = (grid["nz"], grid["nx"])
    if (meta["id"] != row["id"] or meta["case"] != row["case"]
        or meta["split"] != row["legacy_split"]
        or meta["scatter_seed"] != row["scatter_seed"]
        or meta["fs_hz"] != acq["fs_hz"]
        or not np.allclose(meta["angles_deg"], acq["angles_deg"], rtol=0, atol=1e-10)
        or not np.allclose(meta["source_tref_s"], row["source_tref_s"], rtol=0, atol=1e-12)
        or item.rf.shape != tuple(manifest["rf_shape"])
        or item.c.shape != label_shape or item.m.shape != label_shape
        or not all(array.dtype == np.float32 and np.isfinite(array).all()
                   for array in (item.rf, item.c, item.m))):
        raise ValueError(f"Inconsistent RF or labels in {path}")
    return item


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--base-cache", type=Path, default=BASE_CACHE)
    p.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    p.add_argument("--source-roots", nargs="+", type=Path, default=SOURCES)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = build_manifest(args.base_cache, args.supplement, args.source_roots)
    write_json(args.out, manifest)
    print("Legacy splits", Counter(r["legacy_split"] for r in manifest["records"]))
    print("Cases", Counter(r["case"] for r in manifest["records"]))
    print("Variants", Counter(r["variant"] for r in manifest["records"]))


if __name__ == "__main__":
    main()
