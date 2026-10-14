"""Full-year original/optimized/prepared XYZ timing, with bounded block cache.

The prepared path reuses each block across cases/repetitions before advancing
to the next block. Full-year totals sum measured block intervals; this does
not require keeping a whole year's delay cache in RAM. Checks and output I/O
are outside generation timing. See the generated report for timing boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from wall_time import (source, default_lisaorbits, state_from_lisaorbits,
    precompute_jax_link_geometry, eccentric_links_jax, xyz_from_links, jax, np)
from egb_jax_eccentric import prepare_xyz_from_links, pytdi_data_from_links
from pytdi.michelson import compute_factorized_michelson


def original_xyz(state, links):
    """Original bridge, before the GW-only eta shortcut was introduced."""
    data = pytdi_data_from_links(state, links)
    return {channel: compute_factorized_michelson(data, rot=i, order=3,
        delay_order=3, generation=2, unit="frequency")
        for i, channel in enumerate("XYZ")}


def evaluate(mode, orbits, src, times, cached_geometry, cached_tdi):
    start = perf_counter()
    if mode == "prepared":
        geometry = cached_geometry
        geometry_s = 0.
    else:
        state = state_from_lisaorbits(orbits, times)
        geometry = precompute_jax_link_geometry(state)
        geometry_s = perf_counter()-start
    tick = perf_counter()
    links = eccentric_links_jax(src, geometry, batch_size=1,
        physics_mode="1pn_periastron")
    link_s = perf_counter()-tick
    tick = perf_counter()
    if mode == "original":
        xyz = original_xyz(state, links)
    elif mode == "improved":
        xyz = xyz_from_links(state, links, generation=2)
    else:
        xyz = cached_tdi(links)
    tdi_s = perf_counter()-tick
    total_s = perf_counter()-start
    return xyz, np.array([geometry_s, link_s, tdi_s, total_s])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=365.25)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", default="benchmarks/results/pipeline_optimization")
    args = p.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cases = [(0.001, 0.), (0.003, 0.3), (0.01, 0.6)]
    sources = [source(f,e) for f,e in cases]
    modes = ["original", "improved", "prepared"]
    dt = 5.; block = 65536; halo = 128
    count = int(round(args.days*86400/dt))
    blocks = (count+block-1)//block
    sums = np.zeros((len(cases),args.repeats,len(modes),4))
    mismatch_sums = np.zeros((len(cases),args.repeats,2,2))
    setup = np.zeros(2)
    result = {"config": {"cases":cases,"days":args.days,"samples":count,
        "dt_s":dt,"block_samples":block,"halo_samples":halo,"blocks":blocks,
        "repeats":args.repeats,"generation":2,"measurement_order":3,
        "delay_order":3,"physics":"1pn_periastron","fdot":0.,
        "masses_solar":[.6,.4],"batch_size":1,"seed":20260921},
        "platform":platform.platform(),"cpu_count":os.cpu_count(),
        "devices":[str(d) for d in jax.devices()],
        "versions":{p:metadata.version(p) for p in
            ["jax","jaxlib","numpy","scipy","pytdi","lisaorbits"]},
        "source_hashes":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ["src/egb_jax_eccentric/pytdi_bridge.py","src/egb_jax_eccentric/eccentric_jax.py",
             "benchmarks/compare_pipeline_optimizations.py"]},
        "timing_boundary":"Sum of full geometry/links/XYZ block-call wall intervals. Prepared calls exclude cached setup. Checks, imports, JIT warmup and JSON I/O excluded.",
        "cache_scope":"One block reused across all cases and repeats, then released; no whole-year-cache assumption.",
        "rows":[]}
    orbits = default_lisaorbits("equal")
    times = np.arange(-halo,block+halo)*dt
    state = state_from_lisaorbits(orbits,times)
    geometry = precompute_jax_link_geometry(state)
    prepared = prepare_xyz_from_links(state,generation=2)
    warmup = {}
    for mode in modes:
        _,elapsed = evaluate(mode,orbits,sources[1],times,geometry,prepared)
        warmup[mode] = elapsed.tolist()
    result["first_block_warmup_s"] = warmup
    del state,geometry,prepared
    rng = np.random.default_rng(20260921)
    jobs = [(i,r) for i in range(len(cases)) for r in range(args.repeats)]

    def save(completed):
        result["completed_blocks"] = completed
        result["cached_geometry_setup_s"] = float(setup[0])
        result["prepared_tdi_setup_s"] = float(setup[1])
        result["rows"] = []
        for i,(f,e) in enumerate(cases):
            for r in range(args.repeats):
                for m,mode in enumerate(modes):
                    row = {"frequency_hz":f,"eccentricity":e,"repeat":r,"mode":mode}
                    row.update(dict(zip(["geometry_s","links_s","tdi_s","total_s"],sums[i,r,m].tolist())))
                    result["rows"].append(row)
        result["validation"] = []
        for i,(f,e) in enumerate(cases):
            for r in range(args.repeats):
                for m,mode in enumerate(modes[1:]):
                    difference,power = mismatch_sums[i,r,m]
                    result["validation"].append({"frequency_hz":f,"eccentricity":e,
                        "repeat":r,"mode":mode,"relative_l2_xyz":float(np.sqrt(difference/power)) if power else None})
        (output/"timings.json").write_text(json.dumps(result,indent=2)+"\n")

    print("Warmup complete; starting full-year block sweep.",flush=True)
    for bi,start in enumerate(range(0,count,block)):
        keep = min(block,count-start)
        times = (start+np.arange(-halo,block+halo))*dt
        tick = perf_counter()
        state = state_from_lisaorbits(orbits,times)
        geometry = precompute_jax_link_geometry(state)
        setup[0] += perf_counter()-tick
        tick = perf_counter()
        prepared = prepare_xyz_from_links(state,generation=2)
        setup[1] += perf_counter()-tick
        for ji in rng.permutation(len(jobs)):
            i,r = jobs[ji]
            values = {}
            for mi in rng.permutation(len(modes)):
                mode = modes[mi]
                xyz,elapsed = evaluate(mode,orbits,sources[i],times,geometry,prepared)
                sums[i,r,mi] += elapsed
                values[mode] = xyz
            # Compare every generated block, including halo/edge samples.
            for channel in "XYZ":
                reference = values["original"][channel]
                if not np.isfinite(reference).all():
                    raise ValueError("Nonfinite baseline XYZ")
                selected = slice(halo,halo+keep)
                power = np.sum(np.abs(reference[selected])**2)
                for mi,mode in enumerate(modes[1:]):
                    actual = values[mode][channel]
                    if not np.isfinite(actual).all():
                        raise ValueError(f"Nonfinite {mode} XYZ")
                    difference = np.sum(np.abs(actual[selected]-reference[selected])**2)
                    mismatch_sums[i,r,mi] += [difference,power]
                    scale = max(float(np.max(np.abs(reference))),1e-300)
                    if np.max(np.abs(actual-reference))/scale > 1e-12:
                        raise AssertionError(f"Mismatch for {mode}, case {i}, block {bi}")
            del values,xyz
        del state,geometry,prepared
        save(bi+1)
        if (bi+1)%10==0 or bi+1==blocks:
            print(f"Completed {bi+1}/{blocks} full-year blocks across all methods/cases/repeats.",flush=True)
    print("DONE",flush=True)


if __name__ == "__main__":main()
