"""Command-line entry point; run from the repository root."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from .experiment import estimate_resources, run_experiment
from .setup import validate_config


def main():
    parser = argparse.ArgumentParser(description="CPU 2-D dual-frequency array comparison: fullwave data, joint kappa/eta Born inversion.")
    parser.add_argument("--config", type=Path, default=Path("experiments/theranostic_arc/quick.json"))
    parser.add_argument("--out", type=Path, default=Path("out/theranostic_quick"), help="New or empty output directory")
    parser.add_argument("--phantom", type=Path, help="Optional NPZ: x_mm, z_mm, speed_m_s, density_kg_m3 [z,x]")
    parser.add_argument("--data-mode", choices=("fullwave", "born"), help="born is a matched-basis/discretization control")
    parser.add_argument("--grid-refine", type=int, default=1, help="Refine BOTH wave operators, preserving the reconstruction basis (convergence control)")
    parser.add_argument("--forward-refine", type=int, help="Refine only the fullwave data grid relative to the inverse wave grid (mismatch stress test)")
    parser.add_argument("--no-side-receive", action="store_true", help="LF sides are Tx-only: flat/arc observations reduce to central HF")
    parser.add_argument("--replay-fullwave", action="store_true", help="Re-solve recovered media and report nonlinear holdout residuals")
    parser.add_argument("--threads", type=int, default=1, help="BLAS thread limit; 1 prevents sparse-solve oversubscription")
    parser.add_argument("--max-grid-nodes", type=int, default=400000, help="Guard for each wave grid, not a memory estimate")
    parser.add_argument("--max-jacobian-mb", type=float, default=768, help="Guard for retained dense Jacobians in MiB")
    parser.add_argument("--dry-run", action="store_true", help="Print dimensions/resource estimates without running or writing output")
    args = parser.parse_args()
    try:
        if args.grid_refine < 1 or args.threads < 1 or args.max_grid_nodes < 1 or args.max_jacobian_mb <= 0:
            raise ValueError("Refinement, thread count and resource limits must be positive")
        # Apply overrides before the PPW check: --grid-refine may be precisely
        # what makes a custom high-frequency configuration sufficiently sampled.
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or any(not isinstance(config.get(k), dict)
                                               for k in ("grid", "experiment", "geometry")):
            raise ValueError("Configuration requires grid, experiment and geometry objects")
        spacing = config["grid"]["spacing_mm"]
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
            raise ValueError("grid.spacing_mm must be a number")
        config["grid"]["spacing_mm"] /= args.grid_refine
        if args.data_mode:
            config["experiment"]["data_mode"] = args.data_mode
        if args.forward_refine is not None:
            config["experiment"]["forward_refine"] = args.forward_refine
        if args.no_side_receive:
            config["geometry"]["side_receive"] = False
        validate_config(config)
        if config["experiment"]["data_mode"] == "born" and config["experiment"]["forward_refine"] != 1:
            raise ValueError("born requires --forward-refine 1; use fullwave for a discretization mismatch test")
        if args.dry_run:
            print(json.dumps(estimate_resources(config), indent=2, ensure_ascii=False))
            return
        with threadpool_limits(limits=args.threads):
            run_experiment(config, args.out, phantom_path=args.phantom, replay=args.replay_fullwave,
                           max_grid_nodes=args.max_grid_nodes, max_jacobian_mb=args.max_jacobian_mb)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
