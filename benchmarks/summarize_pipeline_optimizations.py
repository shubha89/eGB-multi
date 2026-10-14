"""Summarize and plot the completed end-to-end optimization experiment."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/egb-benchmark-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("benchmarks/results/pipeline_optimization")
data=json.loads((root/"timings.json").read_text())
assert data["completed_blocks"]==data["config"]["blocks"], "Benchmark is incomplete"
cases=data["config"]["cases"]
modes=["original","improved","prepared"]
summary=[]
for f,e in cases:
    row={"frequency_mhz":f*1000,"eccentricity":e}
    for mode in modes:
        selected=[r for r in data["rows"] if r["frequency_hz"]==f and r["eccentricity"]==e and r["mode"]==mode]
        assert len(selected)==data["config"]["repeats"]
        values=[r["total_s"] for r in selected]
        row[f"{mode}_median_s"]=float(np.median(values))
        row[f"{mode}_min_s"]=float(np.min(values))
        row[f"{mode}_max_s"]=float(np.max(values))
        for stage in ["geometry","links","tdi"]:
            row[f"{mode}_{stage}_median_s"]=float(np.median([r[f"{stage}_s"] for r in selected]))
    row["improved_speedup"]=row["original_median_s"]/row["improved_median_s"]
    row["prepared_speedup"]=row["original_median_s"]/row["prepared_median_s"]
    summary.append(row)
with (root/"summary.csv").open("w") as fp:
    writer=csv.DictWriter(fp,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)

fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
x=np.arange(len(cases));width=.24
labels=["Original", "Skip zero-channel work", "Prepared geometry + TDI"]
colors=["#7b8189","#2c6ca6","#288367"]
for i,mode in enumerate(modes):
    med=np.array([r[f"{mode}_median_s"] for r in summary])
    low=med-np.array([r[f"{mode}_min_s"] for r in summary])
    high=np.array([r[f"{mode}_max_s"] for r in summary])-med
    bars=ax.bar(x+(i-1)*width,med,width,color=colors[i],label=labels[i],
        yerr=[low,high],capsize=3,error_kw={"linewidth":1})
    ax.bar_label(bars,labels=[f"{v:.2f} s" for v in med],padding=6,fontsize=10)
ax.set_xticks(x,[f"{f*1000:g} mHz, e={e:g}" for f,e in cases])
ax.set_ylabel("Full-year pipeline wall time [s]")
ax.set_ylim(0,max(r[f"{m}_max_s"] for r in summary for m in modes)*1.23)
ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
setup_for_plot=data["cached_geometry_setup_s"]+data["prepared_tdi_setup_s"]
ax.set_title("Measured XYZ generation: one year at 5 s cadence\nCPU float64; median and range of three warmed repeats\n"
    f"Prepared bars exclude {setup_for_plot:.2f} s of one-time setup",fontsize=11)
fig.legend(*ax.get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False)
fig.savefig(root/"comparison.png",dpi=180)

setup=data["cached_geometry_setup_s"]+data["prepared_tdi_setup_s"]
error=max(v["relative_l2_xyz"] for v in data["validation"])
text=["# Complete-pipeline optimization timings", "",
    "Three source cases, one year (365.25 days), 5 s cadence, 6,311,520 retained samples per channel. CPU float64, fixed 1PN with periastron advance, generation-2 XYZ, interpolation orders 3/3, one source per evaluation. Three warmed repeats per case and method; execution order randomized within blocks.","",
    "| Case | Original | Improved ordinary call | Prepared repeat | Ordinary speedup | Prepared speedup |",
    "|---|---:|---:|---:|---:|---:|"]
for r in summary:
    text.append(f"| {r['frequency_mhz']:g} mHz, e={r['eccentricity']:g} | {r['original_median_s']:.3f} s | {r['improved_median_s']:.3f} s | {r['prepared_median_s']:.3f} s | {r['improved_speedup']:.2f}x | {r['prepared_speedup']:.2f}x |")
text += ["", "## What was timed", "",
    "- Original: geometry + JAX waveform/links + the original full-beatnote pyTDI path, reconstructed explicitly in the benchmark. It does not call the newly optimized ordinary bridge.",
    "- Improved: geometry + JAX waveform/links + the GW-only eta shortcut, with TDI operators rebuilt each call.",
    "- Prepared repeat: JAX waveform/links + prepared TDI evaluation, with geometry and operators reused. Both prepared and ordinary paths still generate a new source waveform on every timed evaluation.",
    "- Every total is the sum of measured complete block-call wall intervals across the year. NumPy materialization synchronizes JAX output. Imports, initial JIT warmup, validation, and output writing are excluded.",
    "", "## Preparation and cache scope", "",
    f"Measured once-per-grid setup totals: geometry **{data['cached_geometry_setup_s']:.3f} s**, TDI operators **{data['prepared_tdi_setup_s']:.3f} s**, combined **{setup:.3f} s**. Add this setup cost when no reusable cache exists; the prepared-repeat column excludes it.",
    "",
    "The experiment prepares each padded 65,536-sample block once, evaluates all three sources and their repetitions, then releases that block. This measures a real bounded-memory reuse workflow; it does not assume that all 97 blocks' prepared operators fit in memory simultaneously. Repeated whole-year likelihood calls would need retained caches, a storage strategy, or repeated setup. Setup is amortized over nine source evaluations in this experiment.",
    "", "## Validation and interpretation", "",
    f"Every generated block was checked against the original, including halo and boundary samples. All outputs were finite. Maximum full-year relative L2 difference across XYZ, cases and repeats: **{error:.3g}**.",
    "",
    "All three modes retain identical source physics, cadence, delay order, TDI generation, and interpolation order. These are complete-pipeline speedups, distinct from the earlier TDI-only 6.3x result. The underlying 5 s configuration is not claimed to be accuracy-converged: the earlier 10 mHz/e=0.6 test changed by about 11% when cadence was halved. The optimizations preserve that configuration's output.",
    "", "![Measured pipeline wall times](comparison.png)", "",
    "[Raw timings, configuration, hashes, and versions](timings.json) · [Stage medians and repeat ranges](summary.csv)", "",
    "Reproduce:", "", "```sh",
    "MPLCONFIGDIR=/private/tmp/egb-benchmark-mpl .venv-benchmark/bin/python benchmarks/compare_pipeline_optimizations.py",
    ".venv-benchmark/bin/python benchmarks/summarize_pipeline_optimizations.py", "```", ""]
(root/"REPORT.md").write_text("\n".join(text))
print(json.dumps({"summary":summary,"setup_s":setup,"max_relative_l2":error},indent=2))
