from keras.src import backend
from keras.src.backend.config import is_nnx_backend_enabled


def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        if x is None:
            x = backend.numpy.ones(())
        for c in x.__class__.__mro__:
            if c.__name__ == "Tracer" and c.__module__.startswith("jax"):
                return True
    return False


if backend.backend() == "jax":
    if is_nnx_backend_enabled():
        from flax import nnx
        jit = nnx.jit
    else:
        import jax
        jit = jax.jit
else:
    # Fallback or placeholder if not JAX backend
    def _identity_jit(func, *args, **kwargs):
        return func
    jit = _identity_jit
