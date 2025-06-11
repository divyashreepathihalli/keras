import os
import sys

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
    def test_sequential_dense_compute_output_spec_with_nnx(self):
        original_backend = os.environ.get("KERAS_BACKEND")
        original_nnx_enabled = os.environ.get("KERAS_NNX_ENABLED")
        original_jax_config = os.environ.get("JAX_CONFIG")

        os.environ["KERAS_BACKEND"] = "jax"
        os.environ["KERAS_NNX_ENABLED"] = "true"
        # Ensure JAX is not in eager debug mode which can affect NNX
        os.environ["JAX_CONFIG"] = "jax_experimental_unsafe_rbg_disable_flag_override=true"

        import importlib

        modules_to_reload = ["keras.src.backend.config", "keras.src.backend", "keras"]
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

        import keras as keras_for_test

        try:
            model = keras_for_test.Sequential([
                keras_for_test.layers.Dense(units=1, input_shape=(10,), name="my_dense_layer")
            ])

            dummy_input_tensor = keras_for_test.KerasTensor(shape=(None, 10), dtype='float32')

            output_spec = model.compute_output_spec(dummy_input_tensor)

            self.assertIsNotNone(output_spec, "output_spec should not be None")
            self.assertEqual(output_spec.shape, (None, 1), f"Expected output shape (None, 1) but got {output_spec.shape}")

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
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
