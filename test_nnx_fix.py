import os
os.environ["KERAS_BACKEND"]="jax"
os.environ["KERAS_NNX_ENABLED"]="true"

import jax
import jax.numpy as jnp
import numpy as np
import keras
# It seems 'from keras.src.backend.jax.layer import JaxLayer' is not needed for the test itself
# and might cause issues if the internal structure changes.
# We are testing keras.Variable behavior with nnx.split.
from flax import nnx

# It's good practice to disable traceback filtering for debugging if needed,
# but for a success test, it might hide useful info if it fails unexpectedly.
# Let's keep it for now as per the plan.
keras.config.disable_traceback_filtering()

try:
    print(f"Keras version: {keras.__version__}")
    print(f"JAX version: {jax.__version__}")
    # Attempt to get flax version, handling cases where nnx might not directly have __version__
    try:
        flax_version = nnx.__version__
    except AttributeError:
        try:
            import flax
            flax_version = flax.__version__
        except (ImportError, AttributeError):
            flax_version = "unknown"
    print(f"Flax version: {flax_version}")
    print(f"KERAS_BACKEND: {os.environ.get('KERAS_BACKEND')}")
    print(f"KERAS_NNX_ENABLED: {os.environ.get('KERAS_NNX_ENABLED')}")
    print(f"keras.config.is_nnx_backend_enabled(): {keras.config.is_nnx_backend_enabled()}")

    variable = keras.Variable(jnp.array(1.0))
    print(f"Type of variable: {type(variable)}")
    print(f"Is variable instance of NnxVariable? {isinstance(variable, keras.src.backend.jax.core.NnxVariable)}")
    print(f"Is variable instance of JaxVariable? {isinstance(variable, keras.src.backend.jax.core.JaxVariable)}")
    # Check if it's a flax nnx Variable directly
    print(f"Is variable instance of nnx.Variable? {isinstance(variable, nnx.Variable)}")


    graphdef, state = nnx.split(variable)
    print("Successfully split Keras variable with nnx.split!")
    print(f"GraphDef type: {type(graphdef)}")
    print(f"State type: {type(state)}")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    # Re-raise to ensure the subtask is marked as failed if an exception occurs
    raise
