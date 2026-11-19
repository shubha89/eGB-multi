import numpy as np
import pytest
from xtdi import LINK_LABELS, prepare_michelson

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from test_michelson import reference_data, reference_xyz


@pytest.mark.parametrize("schedule", ["reference","cpu_channels"])
@pytest.mark.parametrize("generation", [1,2])
@pytest.mark.parametrize("order", [3,25,101])
@pytest.mark.parametrize("unit", ["phase","frequency"])
def test_jax_reference_with_reused_compilation(generation, order, unit, schedule):
    with jax.enable_x64():
        t = np.arange(512)*2.5
        delays = {k: 8.+.1*i+.02*np.sin(t/500+i) for i,k in enumerate(LINK_LABELS)}
        plan = prepare_michelson(t,delays,generation=generation,
                                measurement_order=order)
        backend = plan.to_jax(schedule=schedule)
        values = np.random.default_rng(51).normal(size=(6,t.size))
        links = dict(zip(LINK_LABELS,values))
        expected = reference_xyz(reference_data(t,delays),links,generation,order,7,unit)
        actual = np.asarray(backend.apply_array(jnp.asarray(values),unit=unit))
        assert np.max(np.abs(actual-expected))/np.max(np.abs(expected)) < 1e-12
        # Same executable can be reused for a different orbit plan on this grid.
        second = prepare_michelson(t,{k:v+.01 for k,v in delays.items()},
            generation=generation,measurement_order=order).to_jax(schedule=schedule)
        assert not np.array_equal(np.asarray(second.apply_array(jnp.asarray(values),unit=unit)),actual)


@pytest.mark.parametrize("schedule", ["reference","cpu_channels"])
def test_jax_array_path_batch_and_input_jvp(schedule):
    with jax.enable_x64():
        t = np.arange(256)*2.5
        plan = prepare_michelson(t,np.full((6,t.size),8.3),measurement_order=7).to_jax(schedule=schedule)
        x = jnp.asarray(np.random.default_rng(21).normal(size=(2,6,t.size)))
        v = x*.3
        output, tangent = jax.jvp(plan.apply_array,(x,),(v,))
        np.testing.assert_allclose(np.asarray(tangent),np.asarray(plan.apply_array(v)),atol=3e-14,rtol=1e-12)
        assert output.dtype == jnp.float64
        with pytest.raises(ValueError):
            plan.apply_array(x.astype(jnp.float32))


def test_x64_must_be_explicit():
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match="x64"):
            prepare_michelson(np.arange(128.),np.full((6,128),8.3)).to_jax()



def test_cpu_channel_schedule_composes_with_outer_jit():
    with jax.enable_x64():
        times=np.arange(512)*2.5
        backend=prepare_michelson(times,np.full((6,512),8.3),measurement_order=25).to_jax(schedule="cpu_channels")
        links=jnp.asarray(np.random.default_rng(45).normal(size=(6,512)))
        actual=jax.jit(backend.apply_array)(links)
        direct=np.asarray(backend.apply_array(links))
        expected=reference_xyz(reference_data(times,{label:np.full(512,8.3) for label in LINK_LABELS}),
            dict(zip(LINK_LABELS,np.asarray(links))),2,25,7,"frequency")
        # An outer XLA compilation can change a few float64 rounding bits.
        # Both execution contexts retain the pre-existing 1e-12 oracle gate.
        for result in [np.asarray(actual),direct]:
            assert np.max(np.abs(result-expected))/np.max(np.abs(expected))<1e-12
        assert all(p.device.platform=="cpu" for p in backend.parameters)


def test_cpu_parameter_packing_preserves_ownership_and_exact_tables():
    from xtdi.optimized import prepare_michelson as prepare_compiled
    from xtdi.jax_backend import _owned_stencil_matrix
    from numba import get_num_threads,set_num_threads
    with jax.enable_x64():
        times=np.arange(257)*2.5
        delays=np.array([8.3+.01*np.sin(times/150+i) for i in range(6)])
        threads=get_num_threads();set_num_threads(2)
        try:plan=prepare_compiled(times,delays,measurement_order=25,parallel=True)
        finally:set_num_threads(threads)
        cpu=plan.to_jax(schedule="cpu_channels");reference=plan.to_jax()
        for field in ["centers","fractions","scales"]:
            matrix=_owned_stencil_matrix(plan.operators,field)
            assert not matrix.flags.writeable
            assert np.shares_memory(matrix,getattr(plan.operators[0].stencil,field))
        for a,b in zip(cpu.parameters,reference.parameters,strict=True):
            assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
        with pytest.raises(ValueError,match="schedule"):plan.to_jax(schedule="unknown")


def test_cpu_packing_copies_mutable_or_independent_tables():
    from types import SimpleNamespace
    from xtdi.jax_backend import _owned_stencil_matrix
    original=np.arange(21,dtype=float).reshape(3,7)
    operators=[SimpleNamespace(stencil=SimpleNamespace(fractions=row)) for row in original]
    matrix=_owned_stencil_matrix(operators,"fractions")
    assert not matrix.flags.writeable and not np.shares_memory(matrix,original)
    original[:]=-1.
    np.testing.assert_array_equal(matrix,np.arange(21).reshape(3,7))
