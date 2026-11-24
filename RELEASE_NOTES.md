# eGB-multi 0.1.2.dev0

Time-domain eccentric waveforms now default to full orbit-averaged Peters–Mathews
evolution. Use `evolution_mode="fixed"` in Python or `--evolution fixed` in the
STRIDE and fastlisaresponse examples to reproduce fixed-element runs.

STRIDE is bundled in the normal eGB-multi installation, using the current
JAX-first `lisa_stride` interface and retaining the historical `xtdi` namespace.
The optional `cpu` extra enables its Numba implementation. Detector, source,
sky and ordered-delay calculations retain the qualified development code.
The pyTDI-derived interpolation and stage-graph attribution is retained in
THIRD_PARTY_NOTICES.

`egb_multi.stride` connects the existing eGB source and retarded orbit geometry
to STRIDE without importing research drivers. `examples/egb_stride.py` and
`examples/egb_fastlisaresponse.py` are short installed-package examples.
The optional `fastlisa` extra installs the upstream CPU response package;
CUDA support is installed separately using its upstream instructions.
The fastlisaresponse example includes a tensor-based polarization conversion
and an explicit output-convention map, with a same-input comparison option.

The previously qualified source-phase arithmetic repair is included:
periodic orbital quantities are evaluated before large accumulated angles
round away phase information. The governing source dynamics, initial
conditions and retained waveform physics are unchanged. Independent decimal
phase checks and NumPy/JAX late-epoch tests accompany this update.
The two existing parametric-dynamics references now independently reduce
their input phases with decimal arithmetic before evaluating periodic
functions; their numerical tolerances are unchanged.

This is a development revision, with no new release publication.

# eGB-multi 0.1.1

Library-first source release prepared locally on 22 October 2026.

The 30 canonical eGB-multi modules and nine compatibility modules are unchanged.
User examples and API/physics documentation remain separate from optional tests.
Unit tests, model comparisons, timing studies and paper reproduction are grouped
under `tests/`. Historical referee controllers, planning reports and obsolete
benchmark outputs are retained locally and excluded from this release.

The separate reference-data archive contains 359 current-study evidence files with
unchanged numerical payloads, relocated below `tests/paper/`. It supports the
six current paper figures, 22 nominal annual sources and 14 finite broader-search
comparisons, including the recorded JAXGB/pyEFPE results. Figure replay does not
launch a new simulation, comparison or search.

pyEFPE and fixed-source timing scripts no longer import JAXGB merely to obtain
shared helper functions. Comparison conventions and numerical algorithms are
preserved; no new annual comparison or broad-search campaign is claimed.

PDFs, manuscript drafts, referee correspondence, xTDI development and annual
waveform catalogues are excluded from the Git tree and release assets.
Checksums and scoped validation records accompany the prepared distributions.
Nothing is published by this preparation.
