"""Independent direct link evaluation used by the JAX regression test."""
import numpy as np
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.tdi import LINKS, LINK_LABELS

def rotated_basis(source: EccentricBinaryParams) -> tuple[np.ndarray, np.ndarray]:
    eplus, ecross = source.polarization_basis()
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    return cosps * eplus - sinps * ecross, sinps * eplus + cosps * ecross


def brute_force_eccentric_links(state, sources: list[EccentricBinaryParams]) -> dict[str, np.ndarray]:
    responses = {label: np.zeros(state.t.size, dtype=np.complex128) for label in LINK_LABELS}
    p = state.positions

    for source in sources:
        direction = source.propagation_direction()
        eplus, ecross = rotated_basis(source)
        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
            n = state.unit_arm(sender, receiver)
            length = state.arm_length(sender, receiver)
            denominator = 1.0 - np.einsum("ti,i->t", n, direction)
            delay_sender = state.t - length - np.einsum("ti,i->t", p[:, sender, :], direction)
            delay_receiver = state.t - np.einsum("ti,i->t", p[:, receiver, :], direction)

            hplus_sender, hcross_sender = eccentric_polarizations(source, delay_sender)
            hplus_receiver, hcross_receiver = eccentric_polarizations(source, delay_receiver)
            qplus = np.einsum("ti,ij,tj->t", n, eplus, n)
            qcross = np.einsum("ti,ij,tj->t", n, ecross, n)
            h_sender = qplus * hplus_sender + qcross * hcross_sender
            h_receiver = qplus * hplus_receiver + qcross * hcross_receiver
            responses[label] += 0.5 * (h_sender - h_receiver) / denominator

    return responses
