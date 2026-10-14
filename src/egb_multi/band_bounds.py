"""Cauchy-Schwarz bounds for adding an orthogonal Fourier analysis band."""

import numpy as np


def expanded_band_overlap_upper_bound(in_band_overlap_upper_bound, outside_signal_power_fraction):
    """Bound the expanded-band maximum for the SAME physical template family.

    If eta = ||s_out||^2 / ||s_full||^2 and the in-band family maximum is at
    most U, orthogonality of the two bands gives

        FF_full <= sqrt((1 - eta) * U**2 + eta).

    This maximizes over any relative template power in the two bands. U must
    be a justified upper bound, not merely a saved candidate score. Candidate
    and finite-basin empirical estimates remain explicitly labeled by callers.
    Enlarging the physical template family requires its own search check.
    """
    upper, fraction = float(in_band_overlap_upper_bound), float(outside_signal_power_fraction)
    if not np.isfinite(upper) or not np.isfinite(fraction) or not 0 <= upper <= 1 or not 0 <= fraction <= 1:
        raise ValueError("Finite overlap bound and power fraction in [0,1] required")
    return float(np.sqrt((1-fraction)*upper**2+fraction))


def expanded_band_maximum_increase_bound(in_band_candidate_overlap, outside_signal_power_fraction):
    """Bound FF_full-FF_in for true maxima over the SAME template family.

    A verified candidate supplies L <= FF_in, without supplying an upper
    bound. The Cauchy inequality gives FF_full <= g(FF_in), where
    g(u)=sqrt((1-eta)*u**2+eta). Since g(u)-u decreases on [0,1],

        FF_full-FF_in <= g(L)-L.

    This bounds an increase only, not a decrease, either absolute FF, or a
    maximum over a changed template family. A wider search and a wider band
    remain separate operations. The rational form below avoids cancellation
    when eta is small; eta=1 is interpreted as a limiting bound.
    """
    lower, fraction = float(in_band_candidate_overlap), float(outside_signal_power_fraction)
    if not np.isfinite(lower) or not np.isfinite(fraction) or not 0 <= lower <= 1 or not 0 <= fraction <= 1:
        raise ValueError("Finite candidate overlap and power fraction in [0,1] required")
    denominator = np.sqrt((1-fraction)*lower**2+fraction)+lower
    if denominator == 0:
        return 0.
    return float(fraction*(1-lower**2)/denominator)
