import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.jax.core import NnxVariable


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not keras.config.is_nnx_backend_enabled(),
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

    def test_nnx_variable_shape_and_dtype_provided(self):
        if not keras.config.is_nnx_backend_enabled():
            self.skipTest("NNX backend is not enabled")
        var = keras.src.backend.jax.core.Variable(
            initializer=lambda s, d: jnp.zeros(s, dtype=d),
            shape=(2, 2),
            dtype="int32",
        )
        self.assertEqual(keras.src.backend.common.standardize_dtype(var.dtype), "int32")
        self.assertEqual(var.path, "variable") # Default name

    def test_nnx_variable_shape_provided_dtype_none_uses_floatx(self):
        if not keras.config.is_nnx_backend_enabled():
            self.skipTest("NNX backend is not enabled")
        original_floatx = keras.config.floatx()
        keras.config.set_floatx("float64")
        try:
            var = keras.src.backend.jax.core.Variable(
                initializer=lambda s, d: jnp.zeros(s, dtype=d),
                shape=(2, 2),
                dtype=None,
                name="test_var_float64",
            )
            self.assertEqual(keras.src.backend.common.standardize_dtype(var.dtype), "float64")
            self.assertEqual(var.path, "test_var_float64")
        finally:
            keras.config.set_floatx(original_floatx)

    def test_nnx_variable_shape_none_dtype_provided(self):
        if not keras.config.is_nnx_backend_enabled():
            self.skipTest("NNX backend is not enabled")
        var = keras.src.backend.jax.core.Variable(
            initializer=lambda s, d: jnp.array(0, dtype=d),
            shape=None,
            dtype="bfloat16",
            name="test_var_bfloat16",
        )
        self.assertEqual(keras.src.backend.common.standardize_dtype(var.dtype), "bfloat16")
        self.assertEqual(var.shape, ())
        self.assertEqual(var.path, "test_var_bfloat16")


    def test_nnx_variable_shape_none_dtype_none_uses_floatx(self):
        if not keras.config.is_nnx_backend_enabled():
            self.skipTest("NNX backend is not enabled")
        original_floatx = keras.config.floatx()
        keras.config.set_floatx("float16")
        try:
            var = keras.src.backend.jax.core.Variable(
                initializer=lambda s, d: jnp.array(0, dtype=d),
                shape=None,
                dtype=None,
                name="test_var_float16",
            )
            self.assertEqual(keras.src.backend.common.standardize_dtype(var.dtype), "float16")
            self.assertEqual(var.shape, ())
            self.assertEqual(var.path, "test_var_float16")
        finally:
            keras.config.set_floatx(original_floatx)

    def test_nnx_variable_initializer_defines_dtype(self):
        if not keras.config.is_nnx_backend_enabled():
            self.skipTest("NNX backend is not enabled")
        # This test ensures that KerasVariable's dtype inference from initializer
        # correctly overrides the placeholder's initial floatx dtype.
        initializer = lambda s, d: jnp.array(0, dtype=jnp.int8)
        var = keras.src.backend.jax.core.Variable(
            initializer=initializer,
            shape=(), # Explicitly scalar shape
            dtype=None, # No dtype override
            name="test_var_int8_inferred"
        )
        self.assertEqual(keras.src.backend.common.standardize_dtype(var.dtype), "int8")
        self.assertEqual(var.shape, ())
        self.assertEqual(var.path, "test_var_int8_inferred")
