import os
import pickle # For later pickling test

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.experimental import nnx
import optax # For later optimizer test

from keras.src import backend
from keras.src import initializers as keras_initializers # For later pickling test
from keras.src import layers # For later optimizer test
from keras.src import models # For later optimizer test
from keras.src.backend.common.stateless_scope import StatelessScope # For later pickling test
from keras.src.backend.jax.core import Variable as KerasJaxVariable
from keras.src import testing


# Helper class for Test Case 4 (Adapter approach)
class NNXAdapterModule(nnx.Module):
    def __init__(self, keras_model_or_layer, rngs=None): # Added optional rngs for nnx.Module compatibility
        """
        Initializes the NNXAdapterModule.

        Args:
            keras_model_or_layer: The Keras Model or Layer instance to be wrapped.
            rngs: Optional RNGs for nnx.Module, though not used by this adapter directly.
        """
        self.keras_model = keras_model_or_layer
        # If nnx.Module's __init__ requires specific arguments like 'rngs',
        # they should be passed via super() or handled if this class had its own state.
        # For a simple wrapper, default nnx.Module init is often sufficient.
        # super().__init__(rngs=rngs) # Only if nnx.Module.__init__ expects/requires rngs

    def __call__(self, *args, **kwargs):
        """
        Delegates the call to the underlying Keras model or layer.
        """
        return self.keras_model(*args, **kwargs)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
class JaxCoreVariableTest(testing.TestCase):
    def setUp(self):
        # Ensure JAX backend is set for these tests
        self.original_backend = os.environ.get("KERAS_BACKEND")
        os.environ["KERAS_BACKEND"] = "jax"
        # Reload backend to ensure it picks up the env var if not already set by test runner
        # Note: This might be tricky; Keras often initializes backend at import time.
        # Ideally, the test runner (e.g. pytest) handles backend selection.
        # For now, we assume this or direct KerasJaxVariable import works.
        super().setUp()

    def tearDown(self):
        if self.original_backend:
            os.environ["KERAS_BACKEND"] = self.original_backend
        else:
            del os.environ["KERAS_BACKEND"]
        super().tearDown()

    def test_variable_in_nnx_module(self):
        # Test Case 1: KerasJaxVariable in an nnx.Module
        
        class TestModel(nnx.Module):
            def __init__(self, rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)
                # Ensure initializer is a JAX array for KerasJaxVariable
                self.custom_variable = KerasJaxVariable(
                    initializer=jnp.ones((1, 3), dtype=jnp.float32), 
                    name="custom_var",
                    trainable=True, # Explicitly trainable
                    dtype=jnp.float32 # Explicitly set dtype
                )

            def __call__(self, x):
                return self.linear(x) + self.custom_variable.value

        key = jax.random.key(0)
        # NNX Rngs expects a dictionary of keys
        model = TestModel(rngs={'params': key})

        self.assertTrue(hasattr(model.custom_variable, '_trace_state'))
        
        # Check __repr__ (doesn't crash, produces a string)
        self.assertTrue(isinstance(repr(model.custom_variable), str))
        
        model_state_all = nnx.state(model)
        # Check that custom_variable is part of the state
        self.assertIn('custom_variable', model_state_all)
        self.assertIsInstance(model_state_all['custom_variable'], KerasJaxVariable)

        # Check value
        expected_value = jnp.ones((1, 3), dtype=jnp.float32)
        self.assertSequenceEqual(model.custom_variable.value.shape, expected_value.shape)
        self.assertTrue(jnp.array_equal(model.custom_variable.value, expected_value))

        # Test __jax_array__ implicit conversion
        try:
            added_value = jnp.add(model.custom_variable, 10.0)
            # Expected result of jnp.ones((1,3)) + 10.0
            expected_added_value = jnp.full((1,3), 11.0, dtype=jnp.float32)
            self.assertTrue(jnp.array_equal(added_value, expected_added_value))
        except Exception as e:
            self.fail(f"__jax_array__ test (jnp.add) failed: {e}")

        # Test assignment
        try:
            original_value_tc1 = model.custom_variable.value
            new_assigned_value = jnp.zeros((1, 3), dtype=jnp.float32)
            model.custom_variable.assign(new_assigned_value)
            self.assertTrue(jnp.array_equal(model.custom_variable.value, new_assigned_value))
            # Assign back
            model.custom_variable.assign(original_value_tc1)
            self.assertTrue(jnp.array_equal(model.custom_variable.value, original_value_tc1))
        except Exception as e:
            self.fail(f"Assign test failed: {e}")

    def test_keras_model_with_nnx_optimizer(self):
        # Test Case 2: Keras Sequential model with nnx.Optimizer, using NNXAdapterModule

        # Dummy dataset
        X_train = jnp.arange(100, dtype=jnp.float32).reshape(-1, 1) / 10.0
        Y_train = 2 * X_train - 1 + (jax.random.normal(jax.random.key(0), X_train.shape) * 0.1)

        def dataset(batch_size=10):
            num_samples = X_train.shape[0]
            for i in range(0, num_samples, batch_size):
                yield X_train[i:i + batch_size], Y_train[i:i + batch_size]

        keras_model_instance = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(1,), name="dense_1"),
            layers.Dense(1, name="dense_2"),
        ])

        dummy_input = jnp.ones((1, 1))
        _ = keras_model_instance(dummy_input)  # Build the model
        self.assertTrue(keras_model_instance.built, "Keras model did not build.")
        self.assertTrue(len(keras_model_instance.trainable_weights) > 0, "Keras model has no trainable weights.")

        # Wrap the Keras model with NNXAdapterModule
        # NNXAdapterModule is defined in the same file (core_test.py)
        wrapped_model = NNXAdapterModule(keras_model_instance)
        # No specific rngs passed to adapter, assuming default nnx.Module init is fine

        tx = optax.sgd(1e-3)
        optimizer = nnx.Optimizer(wrapped_model, tx) # Optimizer now uses the wrapped model
        self.assertIsNotNone(optimizer, "nnx.Optimizer failed to create with wrapped model.")

        @nnx.jit
        def train_step_nnx(module_adapter: NNXAdapterModule, opt_state, batch_data):
            x, y = batch_data
            def loss_fn(mdl_adapter: NNXAdapterModule): # Loss function takes the adapter
                y_pred = mdl_adapter(x) # This calls NNXAdapterModule.__call__ -> keras_model(x)
                return jnp.mean((y - y_pred) ** 2)
            
            # nnx.grad operates on the adapter. It should find variables in adapter.keras_model
            grads = nnx.grad(loss_fn, wrt=(nnx.Variable, lambda v: v.trainable))(module_adapter)
            
            # Check if grads are found for the Keras model's variables
            # Grads PyTree will now have 'keras_model' as a top-level key if variables are found there
            found_grads = False
            if 'keras_model' in grads.variables:
                keras_model_grads_state = grads.variables['keras_model'] # This is a State object
                # Iterate through layers of the original keras_model_instance
                for layer in keras_model_instance.layers:
                    if layer.name in keras_model_grads_state.variables: # Check if layer name is a key in grads
                        layer_grads_state = keras_model_grads_state.variables[layer.name]
                        if any(var_name in layer_grads_state.variables for var_name in ['kernel', 'bias']):
                            found_grads = True
                            break
                if found_grads:
                     # Print an example grad norm for debugging if grads are found
                     grad_leaves = jax.tree_util.tree_leaves(grads)
                     # print(f"Debug: Example grad leaf norm: {jnp.linalg.norm(grad_leaves[0]) if grad_leaves else 'N/A'}")
                     pass


            self.assertTrue(found_grads, "No gradients were computed for the Keras model's variables via the adapter.")

            new_opt_state = opt_state.update(grads)
            return module_adapter, new_opt_state

        @nnx.jit
        def test_step_nnx(module_adapter: NNXAdapterModule, batch_data):
            x, y = batch_data
            y_pred = module_adapter(x) # Calls NNXAdapterModule.__call__
            loss = jnp.mean((y - y_pred) ** 2)
            return {'loss': loss}

        current_opt_state = optimizer # Optimizer instance itself is the initial state
        
        initial_loss = float('inf')
        final_loss = float('inf')

        for step, (batch_x, batch_y) in enumerate(dataset()):
            if step >= 5:  # Limit steps for testing
                break
            # Pass the wrapped_model to train_step
            _, current_opt_state = train_step_nnx(wrapped_model, current_opt_state, (batch_x, batch_y))
            if step % 1 == 0:
                logs = test_step_nnx(wrapped_model, (X_train, Y_train))
                loss_value = logs['loss'].item()
                if step == 0:
                    initial_loss = loss_value
                final_loss = loss_value
                self.assertTrue(np.isfinite(loss_value), f"Loss became non-finite at step {step}")
        
        self.assertTrue(np.isfinite(initial_loss), "Initial loss is not finite.")
        self.assertTrue(np.isfinite(final_loss), "Final loss is not finite.")
        self.assertTrue(final_loss <= initial_loss + 1e-1, f"Loss did not decrease or stay stable. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
        print(f"Test test_keras_model_with_nnx_optimizer passed. Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

    def test_keras_jax_variable_pickling(self):
        # Test Case 3: Pickling and Unpickling KerasJaxVariable
        
        original_var_value = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
        
        original_var = KerasJaxVariable(
            initializer=original_var_value,
            name="my_pickled_var",
            trainable=False, # This will map to mutable=False by default in our __init__
            dtype="float32",
            layout=None, # Set to None to simplify test
            mutable=False, # Explicitly passed to nnx.Variable part
            custom_meta_key="custom_meta_value" # Passed to nnx_metadata
        )

        self.assertIn('mutable', original_var._var_metadata)
        self.assertEqual(original_var._var_metadata['mutable'], False)
        self.assertEqual(original_var._var_metadata.get('custom_meta_key'), "custom_meta_value")


        pickled_var_bytes = pickle.dumps(original_var)
        self.assertIsNotNone(pickled_var_bytes, "Pickling returned None.")

        unpickled_var = pickle.loads(pickled_var_bytes)
        self.assertIsInstance(unpickled_var, KerasJaxVariable)

        self.assertEqual(original_var.name, unpickled_var.name)
        self.assertEqual(original_var.path, unpickled_var.path)
        self.assertEqual(original_var.trainable, unpickled_var.trainable)
        self.assertEqual(original_var.dtype, unpickled_var.dtype)
        self.assertSequenceEqual(original_var.shape, unpickled_var.shape)
        self.assertEqual(original_var.ndim, unpickled_var.ndim)
        
        self.assertTrue(jnp.array_equal(original_var.value, unpickled_var.value))
        
        self.assertEqual(original_var._layout, unpickled_var._layout)

        self.assertTrue(hasattr(unpickled_var, '_trace_state'))
        self.assertIsInstance(unpickled_var._trace_state, type(original_var._trace_state))
        
        self.assertTrue(hasattr(unpickled_var, '_var_metadata'))
        self.assertIsInstance(unpickled_var._var_metadata, dict)
        self.assertEqual(unpickled_var._var_metadata.get('mutable'), False)
        self.assertEqual(unpickled_var._var_metadata.get('custom_meta_key'), "custom_meta_value")

        # Confirm it works in JAX operations
        result = jnp.sum(unpickled_var.value + 1)
        expected_result = jnp.sum(original_var_value + 1)
        self.assertEqual(result, expected_result)

        # Test with stateless scope
        with StatelessScope():
            self.assertTrue(jnp.array_equal(unpickled_var.value, original_var_value))
        
        # Test with an initializer
        init_var = KerasJaxVariable(initializer=keras_initializers.Ones(), shape=(2,2), dtype='float32', name="init_var")
        pickled_init_var = pickle.dumps(init_var)
        unpickled_init_var = pickle.loads(pickled_init_var)
        self.assertIsInstance(unpickled_init_var, KerasJaxVariable)
        self.assertTrue(jnp.array_equal(unpickled_init_var.value, jnp.ones((2,2), dtype='float32')))
        self.assertEqual(unpickled_init_var.name, "init_var")
