"""Regressions for batched PM planning and exact-sky response reuse."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.stride import FixedSkyResponse, StreamedDetectorResponse, sampled_pm_batch, sampled_pm_uniform
from lisa_stride import SampledPolarizations
from test_jax_public import basis, fixture


def test_checked_pm_batch_and_fallback(monkeypatch):
    import egb_multi.pm_batch as batch

    calls = []
    single = batch.pm_uniform_polarizations
    def record(*args, **kwargs):
        calls.append(args[0])
        return single(*args, **kwargs)
    monkeypatch.setattr(batch, "pm_uniform_polarizations", record)
    base = EccentricBinaryParams(np.pi*.001, .10, m1_solar=1.1, m2_solar=.4)
    with jax.enable_x64():
        # Sparse samples span one physical year. This checks the PM evolution at
        # those instants and exercises both branches of the integration plan.
        step = 31557600./128
        for sources, fallback in [
                ([base, replace(base, eccentricity=0.), replace(base, m1_solar=1.2)], False),
                ([replace(base, eccentricity=.5, mean_motion=np.pi*.03)], True),
                ([replace(base, t0=-1e6)], True)]:
            calls.clear()
            result = sampled_pm_batch(sources, -3600., step, 129, device="cpu")
            expected = np.stack([sampled_pm_uniform(s, -3600., step, 129).values for s in sources])
            error = np.linalg.norm(result.values-expected, axis=-1)/np.linalg.norm(expected, axis=-1)
            assert np.max(error) < 1e-10
            assert bool(calls) == fallback
        with pytest.raises(ValueError, match="fdot=0"):
            sampled_pm_batch([replace(base, fdot=1e-15)], -3600., step, 129)


def test_fixed_sky_preserves_individual_angles_delays_and_output_order():
    with jax.enable_x64():
        epoch, origin, step = 3e7, 3e7-6000., 5.
        grid = np.arange(4096)*step
        values = jnp.array([np.stack([np.cos(.01*grid+i), .6*np.sin(.017*grid-i)])
                            for i in range(3)])
        angles = jnp.array([.1, -.3, .4])
        stream = StreamedDetectorResponse(epoch+np.arange(289)*5., lambda t: fixture(times=t, retarded=True),
                                         device="cpu", block_samples=128, measurement_order=25)
        sky = basis()
        fixed = FixedSkyResponse.prepare(stream, *sky, origin_s=origin, dt_s=step, samples=len(grid))
        xyz, valid = fixed.apply_array(values, angles)
        assert xyz.shape == (3, 3, 289) and np.all(valid)
        sampled = SampledPolarizations(origin, step, values)
        np.testing.assert_array_equal(fixed.apply(sampled, angles), xyz)
        with pytest.raises(ValueError, match="origin, cadence and order"):
            fixed.apply(replace(sampled, origin_s=origin+step), angles)
        for i in range(3):
            expected = []
            for start in range(0, stream.times.size, stream.block_samples):
                count = min(stream.block_samples, stream.times.size-start)
                det, *_ = stream._entry(start, count)
                part, ok = det.sampled_response_array(values[i], origin, step, *sky, polarization_angle=angles[i])
                assert bool(ok)
                expected.append(part[:, stream.halo:stream.halo+count])
            expected = np.concatenate(expected, axis=-1)
            assert np.max(np.linalg.norm(xyz[i]-expected, axis=-1)/np.linalg.norm(expected, axis=-1)) < 1e-10
        failed, ok = fixed.apply_array(values, angles.at[1].set(jnp.nan))
        assert np.array_equal(ok, [True, False, True]) and np.isnan(failed[1]).all()
        with pytest.raises(ValueError, match="padded"):
            FixedSkyResponse.prepare(stream, *sky, origin_s=epoch+1e6, dt_s=step, samples=len(grid))
        with pytest.raises(ValueError, match="prepared grid"):
            fixed.apply_array(values[..., :-1], angles)
