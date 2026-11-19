"""Shared small-example inputs; no research campaign or private data."""
from pathlib import Path
import argparse
import json
import numpy as np
from egb_multi.eccentric import EccentricBinaryParams


def arguments(description, *, native=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--samples", type=int, default=4096, help="retained output samples")
    parser.add_argument("--dt", type=float, default=2.5, help="source/output cadence in seconds")
    parser.add_argument("--f-mhz", type=float, default=3., help="initial mean motion divided by pi, in mHz")
    parser.add_argument("--eccentricity", type=float, default=.1)
    parser.add_argument("--evolution", choices=["fixed", "peters_mathews"], default="fixed")
    parser.add_argument("--source-backend", choices=["numpy", "jax"], default="jax")
    parser.add_argument("--output", type=Path, help="optional NPZ output; otherwise print a scalar summary only")
    if native:
        parser.add_argument("--backend", choices=["cpu", "cuda11x", "cuda12x"], default="cpu")
        parser.add_argument("--orbit-file", type=Path, help="optional LISA Analysis Tools orbit HDF5")
        parser.add_argument("--orbit-dt", type=float, default=37.5, help="configured orbit interpolation grid in seconds")
        parser.add_argument("--compare-stride", action="store_true", help="same-input, matched instantaneous-geometry check")
    else:
        parser.add_argument("--backend", choices=["jax", "numba"], default="jax")
        parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    if args.samples < 256 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("at least 256 retained samples and a finite positive cadence are required")
    if not np.isfinite(args.f_mhz) or args.f_mhz <= 0 or not 0 <= args.eccentricity < 1:
        parser.error("positive frequency scale and 0 <= eccentricity < 1 required")
    if args.evolution != "fixed" and args.source_backend == "jax":
        parser.error("use --source-backend numpy for Peters-Mathews evolution")
    margin = int(round(5000./args.dt))
    if margin*args.dt != 5000.:
        parser.error("choose a cadence that divides the 5000-second source margin exactly")
    if native and (not np.isfinite(args.orbit_dt) or args.orbit_dt <= 0):
        parser.error("--orbit-dt must be positive and finite")
    return args, margin


def inputs(args, margin):
    source = EccentricBinaryParams(
        mean_motion=np.pi*args.f_mhz*1e-3, eccentricity=args.eccentricity,
        m1_solar=1.1, m2_solar=.4, inclination=.9, phi0=.7,
        beta=.3, lambda_=1.2, psi=.4, t0=0.,
    )
    full_times = np.arange(args.samples+2*margin, dtype=np.float64)*args.dt
    keep = slice(margin, margin+args.samples)
    return source, full_times, full_times[keep], keep


def finish(args, times, xyz, summary):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3, args.samples) or not np.all(np.isfinite(xyz)):
        raise ValueError("invalid XYZ output")
    summary.update(samples=args.samples, dt_s=args.dt,
                   retained_start_s=float(times[0]), source_origin_s=0.,
                   channels="X,Y,Z; real fractional frequency",
                   xyz_norm=float(np.linalg.norm(xyz)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, times=times, xyz=xyz, metadata=json.dumps(summary))
        summary["saved_output"] = str(args.output)
    print(json.dumps(summary, indent=2))
