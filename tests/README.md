# Tests and optional research workflows

The library and its normal installation are independent of this directory.
All package tests, model comparisons, timing studies and paper-reproduction
programs are grouped here.

| Directory | Purpose | Runs in the default pytest command? |
| --- | --- | --- |
| `unit/` | Library regressions and safe reference-data restoration | Yes |
| `helpers/` | Independent direct-link reference used by a unit test | Imported by that test |
| `comparisons/` | Optional JAXGB and pyEFPE comparisons | No |
| `benchmarks/` | Explicit source and prepared-response timing studies | No |
| `paper/` | Restore and verify saved evidence; reproduce six paper figures | No |

```sh
python -m pip install ".[orbits,analysis,test,precision]"
python -m pytest
```

The pinned macOS arm64/Python 3.13 reference environment is recorded in
`constraints-macos-arm64-py313.txt`. Dependency ranges on other platforms need
their own numerical checks; exact versions do not guarantee bitwise agreement.

[Model comparison instructions](comparisons/README.md),
[timing instructions](benchmarks/README.md), and
[paper reproduction](paper/README.md) describe the optional workflows.
`python tests/smoke.py --require-wheel-install` checks a short installed native
response and its circular spectral counterpart. It does not run an annual study.

Historical referee campaign controllers, scheduling experiments, planning notes
and obsolete benchmark outputs are retained in the local project archive and
are excluded from this release. The retained library tests have not been replaced
by a smaller numerical model. Scientific data and the 39 library/compatibility
module payloads are unchanged by the directory cleanup.
