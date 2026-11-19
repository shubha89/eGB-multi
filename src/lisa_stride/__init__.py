"""STRIDE: JAX-first sampled LISA response with an optional Numba CPU backend.

Enable JAX float64 before preparation: jax.config.update('jax_enable_x64',True).
The xtdi namespace preserves the established explicit CPU API.
"""
from xtdi.common import LINK_LABELS
from xtdi.jax_response import JAXDetectorResponse,JAXStreamedDetectorResponse,JAXTDIPlan
from .sampled import SampledPolarizations
__version__='0.1.1.dev2'


def _cpu(module,name):
    from importlib import import_module
    try:return getattr(import_module(module),name)
    except ModuleNotFoundError as exc:
        if exc.name=='numba':
            raise ModuleNotFoundError('Numba CPU backend requires pip install "egb-multi[cpu]"') from exc
        raise


def _backend(backend):
    if backend not in ('jax','numba'):raise ValueError('backend must be jax or numba')
    return backend


class DetectorResponse:
    """Factory: prepare Cartesian geometry with JAX by default."""
    @staticmethod
    def prepare(geometry,*,backend='jax',**options):
        if _backend(backend)=='jax':return JAXDetectorResponse.prepare(geometry,**options)
        return _cpu('xtdi.inference','DetectorResponse').prepare(geometry,**options)


def StreamedDetectorResponse(times,geometry_provider,*,backend='jax',**options):
    """Bounded detector cache; a new physical sky is evaluated on every call."""
    if _backend(backend)=='jax':return JAXStreamedDetectorResponse(times,geometry_provider,**options)
    return _cpu('xtdi.inference','StreamedDetectorResponse')(times,geometry_provider,**options)


class TDIPlan:
    """Factory: ordered six-link TDI, independent of any waveform generator."""
    @staticmethod
    def prepare(times,delays_s,*,backend='jax',**options):
        options.setdefault('measurement_order',25)
        options.setdefault('delay_order',7)
        if _backend(backend)=='jax':return JAXTDIPlan.prepare(times,delays_s,**options)
        return _cpu('xtdi.optimized','prepare_michelson')(times,delays_s,**options)


xyz_to_aet=JAXDetectorResponse.xyz_to_aet
__all__=['LINK_LABELS','DetectorResponse','StreamedDetectorResponse','TDIPlan',
         'SampledPolarizations','JAXDetectorResponse','JAXStreamedDetectorResponse',
         'JAXTDIPlan','xyz_to_aet']
