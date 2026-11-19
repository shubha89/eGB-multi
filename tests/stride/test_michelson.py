"""Reference equivalence, nested-rate checks, and independent laser limit."""

import numpy as np
import pytest

from xtdi import LINK_LABELS, prepare_michelson, xyz_to_aet

pytdi = pytest.importorskip("pytdi")
from pytdi.core import LISATDICombination
from pytdi.michelson import compute_factorized_michelson


def reference_data(t, delays, rates=None):
    data = pytdi.Data({}, {"d_"+k: v for k, v in delays.items()}, 1/(t[1]-t[0]))
    if rates is None:
        data.compute_delay_derivatives()
    else:
        data.delay_derivatives = {"d_"+k: v for k, v in rates.items()}
    return data


def reference_xyz(data, links, generation, order, delay_order, unit="frequency"):
    etas = {"eta_"+label: links[label] for label in LINK_LABELS}
    return np.array([compute_factorized_michelson(data, etas=etas, rot=i,
        order=order, delay_order=delay_order, generation=generation, unit=unit)
        for i in range(3)])


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("order", [3, 45, 101])
@pytest.mark.parametrize("varying", [False, True])
def test_full_pytdi_equivalence_and_plan_reuse(generation, order, varying):
    t = 123456. + np.arange(1536)*2.5
    delays = {label: np.full(t.size, 8.1 + .09*i) +
              (0.03*np.sin((t-t[0])/1500+i) if varying else 0)
              for i, label in enumerate(LINK_LABELS)}
    plan = prepare_michelson(t, delays, generation=generation,
                            measurement_order=order, delay_order=7)
    data = reference_data(t, delays)
    rng = np.random.default_rng(432)
    for _ in range(2):
        links = {label: rng.normal(size=t.size) for label in LINK_LABELS}
        reference = reference_xyz(data, links, generation, order, 7)
        actual = plan(links)
        # Include endpoints, not merely the interior where differences hide.
        assert np.max(np.abs(actual-reference)) / np.max(np.abs(reference)) < 1e-12
    assert plan.nbytes < 1100*t.size
    # A changed grid or detector must produce a different plan identity.
    changed = dict(delays); changed["12"] = delays["12"] + 1e-4
    assert prepare_michelson(t, changed, generation=generation,
        measurement_order=order).fingerprint != plan.fingerprint


def test_nested_shifts_rates_and_intermediate_arms():
    t = np.arange(1024)*2.5
    delays = {label: 8.2 + .04*i + .01*np.sin(t/1200+i)
              for i, label in enumerate(LINK_LABELS)}
    rates = {label: .01/1200*np.cos(t/1200+i)
             for i, label in enumerate(LINK_LABELS)}
    data = reference_data(t, delays, rates)
    plan = prepare_michelson(t, delays, delay_derivatives=rates)
    for operator in plan.operators:
        combination = LISATDICombination({"x": [(1, ["D_"+k for k in operator.chain])]})
        shifts, derivatives = combination.build_shifts(**data.args, order=7)
        key = tuple("D_"+k for k in operator.chain)
        np.testing.assert_allclose(operator.shift_s, shifts[key], rtol=1e-14, atol=1e-13)
        np.testing.assert_allclose(operator.rate, derivatives[key], rtol=1e-13, atol=1e-18)
    links = {label: np.sin(.02*t+i) for i, label in enumerate(LINK_LABELS)}
    _, stages = plan.apply(links, return_stages=True)
    etas = {"eta_"+k: v for k, v in links.items()}
    a = LISATDICombination({"eta_12": [(1, [])], "eta_21": [(1, ["D_12"])]}).build(**data.args, order=7)(etas, order=101)
    b = LISATDICombination({"eta_13": [(1, [])], "eta_31": [(1, ["D_13"])]}).build(**data.args, order=7)(etas, order=101)
    np.testing.assert_allclose(stages["X.arms"], (a, b), atol=2e-14, rtol=1e-12)
    rounds = {"arm_a": a, "arm_b": b}
    r1 = LISATDICombination({"arm_a": [(1, [])], "arm_b": [(1, ["D_12", "D_21"])]}).build(**data.args, order=7)(rounds, order=101)
    r2 = LISATDICombination({"arm_a": [(1, ["D_13", "D_31"])], "arm_b": [(1, [])]}).build(**data.args, order=7)(rounds, order=101)
    np.testing.assert_allclose(stages["X.rounds"], (r1, r2), atol=3e-14, rtol=1e-12)


@pytest.mark.parametrize("unit", ["phase", "frequency"])
def test_batch_and_aet(unit):
    t = np.arange(512)*5.
    delays = {label: 8. + .1*i + .001*np.sin(t/100+i) for i, label in enumerate(LINK_LABELS)}
    plan = prepare_michelson(t, delays, measurement_order=7)
    inputs = np.random.default_rng(98).normal(size=(3, 6, 512))
    actual = plan.apply(inputs, unit=unit)
    for i in range(3):
        expected = reference_xyz(reference_data(t, delays), dict(zip(LINK_LABELS, inputs[i])), 2, 7, 7, unit)
        np.testing.assert_allclose(actual[i], expected, rtol=1e-11, atol=2e-13)
    aet = xyz_to_aet(actual)
    assert aet.shape == (3, 3, 512)
    np.testing.assert_allclose(np.sum(aet*aet), np.sum(actual*actual), rtol=1e-14)


@pytest.mark.parametrize("generation", [1, 2])
def test_static_commuting_laser_cancellation(generation):
    # Integer delays are exact independently of the interpolation oracle.
    dt, size = 2.5, 2048
    t = np.arange(size)*dt
    counts = {label: 2+i for i, label in enumerate(LINK_LABELS)}
    delays = {label: np.full(size, counts[label]*dt) for label in LINK_LABELS}
    lasers = np.random.default_rng(19).normal(size=(3, size))*1e-13
    links = {}
    for label in LINK_LABELS:
        receiver, emitter = int(label[0])-1, int(label[1])-1
        k = counts[label]
        delayed = np.pad(lasers[emitter], (k, 0))[:size]
        links[label] = delayed - lasers[receiver]
    actual = prepare_michelson(t, delays, generation=generation)(links)
    halo = 8*max(counts.values())
    assert np.max(np.abs(actual[:, halo:-halo])) < 2e-27


def test_zero_delay_and_invalid_inputs():
    t = np.arange(128.)
    zero = np.zeros((6, t.size))
    plan = prepare_michelson(t, zero)
    np.testing.assert_array_equal(plan(np.ones((6, t.size))), np.zeros((3, t.size)))
    with pytest.raises(ValueError):
        plan(np.zeros((6, 127)))
    with pytest.raises(ValueError):
        prepare_michelson(t, zero-1.)
    with pytest.raises(ValueError):
        plan({"sci_12": np.ones(t.size)})
    with pytest.raises(ValueError):
        plan.apply(zero, unit="strain")
    with pytest.raises(ValueError):
        prepare_michelson(t[::-1], zero)
