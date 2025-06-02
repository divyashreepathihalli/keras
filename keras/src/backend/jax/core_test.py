import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.jax.core import Variable as KerasJaxVariable


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
class JaxCoreVariableTest(testing.TestCase):
    def setup(self):
        super().setup()

        class NNXModel(nnx.Module):
            def __init__(self, rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)
                self.custom_variable = KerasJaxVariable(jnp.ones((1, 3)))

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
        # 1. Keras backend is set to JAX by the class decorator @pytest.mark.skipif
        # 2. Necessary imports are jax, jax.numpy as jnp, keras, flax.nnx
        
        # 3. Create a Keras JAX variable
        variable = keras.Variable(jnp.array(1.0), name="test_var")

        # 4. Split the variable using nnx.split()
        # nnx.split expects a Module, so we wrap the variable in a simple one
        class SimpleModule(nnx.Module):
            def __init__(self):
                self.v = variable
        
        module_instance = SimpleModule()
        graphdef, state = nnx.split(module_instance)

        # 5. Modify the state
        # The state will be a dict {'v': {'raw_value': array(1., dtype=float32)}}
        # or similar, depending on NNX internal structure for Keras Variables.
        # We need to target the actual array value.
        def update_fn(x):
            if isinstance(x, jnp.ndarray):
                return x + 1
            return x
        
        modified_state = jax.tree.map(update_fn, state)

        # 6. Merge the state back
        # variable2 will be the module instance with updated state
        module_instance_2 = nnx.merge(graphdef, modified_state)
        variable2 = module_instance_2.v # Get the variable from the module

        # 7. Assert that variable2 is the same instance as variable
        self.assertIs(variable2, variable)

        # 8. Assert that the internal _value and the public value property are consistent
        self.assertEqual(variable2._value, variable2.value)

        # 9. Assert that the value is the updated one
        self.assertAllClose(variable2.value, jnp.array(2.0))
        self.assertAllClose(variable._value, jnp.array(2.0)) # Also check original variable's _value
