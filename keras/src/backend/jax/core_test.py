import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.config import is_nnx_backend_enabled

if is_nnx_backend_enabled():
    from flax import nnx

    from keras.src.backend.jax.core import NnxVariable


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not is_nnx_backend_enabled(),
    reason="Test requires NNX backend to be enabled by default for setup.",
)
class JaxCoreVariableTest(testing.TestCase):
    def setup(self):
        super().setup()

        class NNXModel(nnx.Module):
            def __init__(self, rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)
                # Use NnxVariable directly as KerasJaxVariable
                # might be JaxVariable if NNX is disabled globally.
                self.custom_variable = NnxVariable(jnp.ones((1, 3)))

            def __call__(self, x):
                return self.linear(x) + self.custom_variable

        self.nnx_model = NNXModel(rngs=nnx.Rngs(0))
        self.keras_nnx_model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=(10,))]
        )
        self.single_dummy_input = np.random.rand(1, 10)

    def test_variable_in_nnx_module(self):
        self.assertTrue(hasattr(self.nnx_model.custom_variable, "_trace_state"))
        self.assertIsNotNone(self.nnx_model.custom_variable._trace_state)
        self.assertAllEqual(self.nnx_model.custom_variable.value, [[1, 1, 1]])
        self.assertTrue(
            isinstance(self.nnx_model.custom_variable, nnx.Variable)
        )

    def test_model_saving(self):
        path = os.path.join(self.get_temp_dir(), "model.keras")
        original_outputs = self.keras_nnx_model(self.single_dummy_input)
        self.keras_nnx_model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)
        restored_outputs = restored_model(self.single_dummy_input)
        self.assertAllEqual(original_outputs, restored_outputs)

    def test_keras_variable_nnx_split_merge_sync(self):
        variable1 = keras.Variable(jnp.array(1.0))
        graphdef, state = nnx.split(variable1)
        state = jax.tree.map(lambda x: x + 1, state)
        variable2 = nnx.merge(graphdef, state)
        self.assertEqual(variable2._value, variable2.value)


class JAXCoreTest(testing.TestCase):
    def test_sequential_dense_fit_with_nnx(self):
        original_backend = os.environ.get("KERAS_BACKEND")
        original_nnx_enabled = os.environ.get("KERAS_NNX_ENABLED")
        original_jax_config = os.environ.get("JAX_CONFIG")

        os.environ["KERAS_BACKEND"] = "jax"
        os.environ["KERAS_NNX_ENABLED"] = "true"
        # Ensure JAX is not in eager debug mode which can affect NNX
        os.environ["JAX_CONFIG"] = "jax_experimental_unsafe_rbg_disable_flag_override=true"


        import importlib
        import keras as keras_reload
        from keras.src import backend as backend_reload

        # Reloading the config and backend modules forces Keras to re-evaluate
        # the environment variables for backend selection.
        importlib.reload(keras_reload.src.backend.config)
        importlib.reload(backend_reload) # Reload specific backend module
        importlib.reload(keras_reload) # Reload main Keras

        try:
            # Model definition
            model = keras_reload.Sequential([
                keras_reload.layers.Dense(units=1, input_shape=(10,), name="my_dense_layer")
            ])

            # Dummy data
            np.random.seed(42)
            num_samples = 10
            input_features = 10
            X_dummy = np.random.rand(num_samples, input_features).astype(np.float32)
            y_dummy = (X_dummy[:, 0] + X_dummy[:, 1] * 0.5 +
                       np.random.randn(num_samples) * 0.1)
            y_dummy = y_dummy.reshape(-1, 1).astype(np.float32)

            # Compile
            model.compile(optimizer=keras_reload.optimizers.SGD(learning_rate=0.01),
                          loss='mean_squared_error')

            # Fit (the critical part)
            history = model.fit(X_dummy, y_dummy, epochs=2, batch_size=4, verbose=0)

            self.assertIsNotNone(history, "Model fitting should return a history object.")
            self.assertIn('loss', history.history, "History object should contain loss.")
            self.assertEqual(len(history.history['loss']), 2, "Should have run for 2 epochs.")

        finally:
            # Cleanup environment variables
            if original_backend:
                os.environ["KERAS_BACKEND"] = original_backend
            else:
                if "KERAS_BACKEND" in os.environ:
                    del os.environ["KERAS_BACKEND"]

            if original_nnx_enabled:
                os.environ["KERAS_NNX_ENABLED"] = original_nnx_enabled
            else:
                if "KERAS_NNX_ENABLED" in os.environ:
                    del os.environ["KERAS_NNX_ENABLED"]

            if original_jax_config:
                os.environ["JAX_CONFIG"] = original_jax_config
            else:
                if "JAX_CONFIG" in os.environ:
                    del os.environ["JAX_CONFIG"]

            # Important: Reload Keras again to revert to the original backend settings
            # for subsequent tests in the same test suite.
            importlib.reload(keras_reload.src.backend.config)
            importlib.reload(backend_reload)
            importlib.reload(keras_reload)
