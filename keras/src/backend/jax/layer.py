from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    class NnxLayer(nnx.Module):
        pass

    # For type checking
    class JaxLayer:
        pass

else:

    class JaxLayer:
        pass

    # For type checking
    class NnxLayer:
        pass
