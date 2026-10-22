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
