from keras.src import backend
from keras.src.backend.config import is_nnx_backend_enabled
import functools # Added for functools.wraps

def is_in_jax_tracing_scope(x=None):
    if backend.backend() == "jax":
        if x is None:
            x = backend.numpy.ones(())
        for c in x.__class__.__mro__:
            if c.__name__ == "Tracer" and c.__module__.startswith("jax"):
                return True
    return False

def jit(func=None, *args, **kwargs):
    """
    A JIT decorator that defers backend checking until call time
    to avoid circular imports.
    This function can be used as a decorator directly (@jit) or
    as a decorator factory (@jit(static_argnums=...)).
    """
    if func is None:
        # Called as @jit(...)
        return lambda f: _jit_wrapper(f, *args, **kwargs)
    else:
        # Called as @jit
        return _jit_wrapper(func, *args, **kwargs)

def _jit_wrapper(func, *decorator_args, **decorator_kwargs):
    # We need to access the backend at runtime here
    current_backend = backend.backend() # Deferred backend check

    if current_backend == "jax":
        if is_nnx_backend_enabled():
            from flax import nnx
            return nnx.jit(func, *decorator_args, **decorator_kwargs)
        else:
            import jax
            return jax.jit(func, *decorator_args, **decorator_kwargs)
    else:
        # Fallback for non-JAX backends
        if decorator_args or decorator_kwargs:
            # If jit had arguments, the identity needs to handle them
            # The decorated function `func` itself doesn't receive decorator_args/kwargs
            @functools.wraps(func)
            def identity_wrapper(*call_args, **call_kwargs):
                return func(*call_args, **call_kwargs)
            return identity_wrapper
        return func
