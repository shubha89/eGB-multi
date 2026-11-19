"""Legacy CPU API, imported lazily. Use lisa_stride for the JAX-first API."""
from importlib import import_module
from .common import LINK_LABELS
__version__ = "0.1.1.dev2"
_MODULES = {
    "FractionalDelay":"delay", "timeshift":"delay",
    "MichelsonPlan":"michelson", "prepare_michelson":"michelson",
    "xyz_to_aet":"michelson", "StreamedMichelson":"streaming",
    "DetectorResponse":"inference", "StreamedDetectorResponse":"inference",
}
__all__ = ["LINK_LABELS", *_MODULES]
def __getattr__(name):
    if name not in _MODULES: raise AttributeError(name)
    try: value=getattr(import_module("."+_MODULES[name],__name__),name)
    except ModuleNotFoundError as exc:
        if exc.name == "numba":
            raise ModuleNotFoundError("Legacy CPU APIs require egb-multi[cpu] (Numba)") from exc
        raise
    globals()[name]=value
    return value
