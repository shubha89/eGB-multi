"""Keep the bundled JAX contract runnable without optional Numba."""
import importlib.util

if importlib.util.find_spec("numba") is None:
    collect_ignore = [
        "test_delay.py", "test_egb_adapter.py", "test_inference.py",
        "test_jax_backend.py", "test_jax_response.py", "test_michelson.py",
        "test_optimized_interpolation.py", "test_optimized_links.py",
        "test_optimized_michelson.py", "test_optimized_preparation.py",
        "test_sampled.py", "test_streaming.py",
    ]
