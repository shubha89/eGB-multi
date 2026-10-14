"""Protect the scientific completion gate against skipped optimizer work."""

import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from referee_task_2_optimizer_certification import restart_coverage_complete


def test_empty_partial_or_duplicated_restarts_do_not_pass_validation():
    complete=[{'basin_index':basin,'seed':seed} for basin in range(3) for seed in (7,9)]
    assert restart_coverage_complete(complete,3,[7,9])
    assert not restart_coverage_complete([],3,[7,9])
    assert not restart_coverage_complete(complete[:-1],3,[7,9])
    assert not restart_coverage_complete(complete+[complete[0]],3,[7,9])
    assert not restart_coverage_complete([],0,[])
