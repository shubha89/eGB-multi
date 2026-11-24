"""Empirical cadence selection for one Julian year of PM-on eGB + STRIDE."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import math
from numbers import Real


YEAR_S = 31_557_600


@lru_cache(maxsize=1)
def _calibration():
    resource = files("egb_multi").joinpath("data", "annual_pm_cadence.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _bounds(value, axis):
    if isinstance(value, Real) and not isinstance(value, bool):
        low = high = float(value)
    else:
        try:
            low, high = value
        except (TypeError, ValueError) as error:
            raise ValueError("Supply one value or a pair of bounds.") from error
    if not all(isinstance(x, Real) and not isinstance(x, bool) for x in (low, high)):
        raise ValueError("Bounds must be real numbers.")
    if not axis[0] <= low <= high <= axis[-1]:
        raise ValueError(f"Bounds must be ordered and within [{axis[0]}, {axis[-1]}].")
    return low, high


@dataclass(frozen=True)
class CadenceChoice:
    cadence_s: float | None
    measured_point: bool
    checked_points: tuple[str, ...]


def recommend_cadence(frequency_mhz, eccentricity, *, criterion="waveform",
                      snr=100., residual_limit=1.):
    """Select a native grid for a point or rectangular prior in (mean_motion/pi, e0).

    Frequency is in mHz. 'waveform' requires annual XYZ errors below 1e-8
    in each channel. 'likelihood' requires the empirical A/E residual SNR
    to be below residual_limit at the specified signal SNR. An unresolved
    region returns cadence_s=None. Interior choices require validation for
    the intended population, sky, orbit and noise model.
    """
    if criterion not in ("waveform", "likelihood"):
        raise ValueError("criterion must be 'waveform' or 'likelihood'")
    if not all(math.isfinite(x) and x > 0 for x in (snr, residual_limit)):
        raise ValueError("snr and residual_limit must be finite and positive")
    data = _calibration()
    fs, es = data["frequency_mHz"], data["eccentricity"]
    f0, f1 = _bounds(frequency_mhz, fs)
    e0, e1 = _bounds(eccentricity, es)
    points = data["points"]
    exact = [name for name, row in points.items()
             if f0 == f1 == row["frequency_mhz"] and e0 == e1 == row["eccentricity"]]
    flo, fhi = fs[bisect_right(fs, f0)-1], fs[bisect_left(fs, f1)]
    elo, ehi = es[bisect_right(es, e0)-1], es[bisect_left(es, e1)]
    anchors = exact or [name for name, row in points.items()
                        if flo <= row["frequency_mhz"] <= fhi and elo <= row["eccentricity"] <= ehi]
    choices = []
    for name in anchors:
        row = points[name]
        if criterion == "waveform":
            choices.append(set(row["waveform_s"]))
        else:
            passing = set()
            for step, error in sorted(row["ae_relative_l2"].items(), key=lambda item: float(item[0])):
                if error * snr >= residual_limit:
                    break
                passing.add(float(step))
            choices.append(passing)
    return CadenceChoice(max(set.intersection(*choices), default=None), bool(exact), tuple(anchors))


@dataclass(frozen=True)
class AnnualPMGrid:
    """Half-open, native observation grid. Source epochs are barycentric seconds."""
    cadence_s: float
    samples: int

    def sampled_source(self, source, *, start_s=0., padding_s, **options):
        """Render PM polarizations including the requested padding on each side.

        start_s is the first observation sample in the source's time coordinate.
        Shift the returned origin by the same epoch offset as the detector when
        its timestamps are expressed in a different time coordinate.
        """
        if not math.isfinite(padding_s) or padding_s < 0 or not math.isfinite(start_s):
            raise ValueError("Supply a finite start and nonnegative finite padding in seconds.")
        from .stride import sampled_pm_uniform
        pad = math.ceil(padding_s / self.cadence_s)
        return sampled_pm_uniform(source, start_s-pad*self.cadence_s, self.cadence_s,
                                  self.samples+2*pad, **options)


def annual_pm_grid(source, *, cadence_s=2.5, snr=100., residual_limit=1.):
    """Build an annual grid at an explicit cadence or choose it from the PM map.

    'waveform' and 'likelihood' opt into their corresponding measured rules.
    For a fixed PE prior, use recommend_cadence on its bounds once and supply
    that numerical cadence for every proposal. The final sample precedes 1 yr.
    """
    if isinstance(cadence_s, str):
        f = source.mean_motion / math.pi * 1000.
        for node in set(r["frequency_mhz"] for r in _calibration()["points"].values()):
            if abs(f-node) <= math.ulp(node):
                f = node
                break
        choice = recommend_cadence(f, source.eccentricity, criterion=cadence_s,
                                   snr=snr, residual_limit=residual_limit)
        if choice.cadence_s is None:
            raise ValueError("No calibrated cadence passed for this source and criterion.")
        cadence_s = choice.cadence_s
    if (isinstance(cadence_s, bool) or not isinstance(cadence_s, Real)
            or not math.isfinite(cadence_s) or cadence_s <= 0):
        raise ValueError("cadence_s must be positive and finite")
    return AnnualPMGrid(float(cadence_s), math.ceil(YEAR_S/cadence_s))
