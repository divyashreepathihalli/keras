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

        # 4. Split the variable directly using nnx.split()
        graphdef, state = nnx.split(variable)

        # 5. Modify the state
        # When splitting a single nnx.Variable, the state might be the value itself
        # or a dict like {'value': ..., 'metadata': ...}.
        # jax.tree.map will traverse this and update any jnp.ndarray leaves.
        def update_fn(x):
            if isinstance(x, jnp.ndarray): # Or jax.Array for more generality
                return x + 1
            return x
        
        modified_state = jax.tree.map(update_fn, state)

        # 6. Merge the state back
        # variable2 should be the same instance as the original variable, updated.
        variable2 = nnx.merge(graphdef, modified_state)

        # 7. Assert that variable2 is the same instance as variable (critical check first)
        self.assertIs(variable2, variable)

        # Assert that variable2._value is immediately updated after merge and before other accesses.
        # variable2._value now calls the property getter which reads raw_value.
        self.assertAllClose(variable2._value, jnp.array(2.0), msg="variable2._value should be immediately updated after merge")

        # 8. Assert that the internal _value (property) and the public value property are consistent
        self.assertEqual(variable2._value, variable2.value, msg="_value and .value should be consistent")

        # 9. Assert that the public value property reflects the update
        self.assertAllClose(variable2.value, jnp.array(2.0), msg="variable2.value should reflect the update")
        # self.assertAllClose(variable._value, jnp.array(2.0)) # This is redundant due to assertIs and the immediate check above.
