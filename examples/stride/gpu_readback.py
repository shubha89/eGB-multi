"""Optional blocking CUDA-to-NumPy readback; response arithmetic remains JAX.

Requires a CuPy installation matching the JAX CUDA runtime. It is not imported
by STRIDE's core and is unnecessary for a likelihood that stays on the GPU.
The DLPack view shares the existing device buffer; asnumpy then copies its
unchanged bytes into CPU memory. No polarization/response calculation is done.
"""


def numpy_from_cuda(array):
    """Copy a concrete, single-device JAX CUDA array to a NumPy array.

    Validate the response's separate validity flag before accepting its data.
    This copy preserves NaNs and dtype; it does not repair invalid responses.
    """
    if not hasattr(array, '__dlpack_device__') or array.__dlpack_device__()[0] != 2:
        raise ValueError('a concrete single-device CUDA array is required')
    import cupy as cp
    return cp.asnumpy(cp.from_dlpack(array))
