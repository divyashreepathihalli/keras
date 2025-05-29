import os
os.environ["KERAS_BACKEND"] = "jax" # Ensure JAX backend is set

import jax
import numpy as np
import keras
# JaxLayer is implicitly used when KERAS_BACKEND is jax
# from keras.src.backend.jax.layer import JaxLayer 
from flax.experimental import nnx # User's script used experimental

# --- Start of user-provided script ---
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(10,))
])

print("Model created successfully")

def mse_loss(y_true, y_pred):
    # Ensure y_pred is not a tuple if model returns (output, new_state)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    return jax.numpy.mean((y_true - y_pred) ** 2)

optimizer = keras.optimizers.SGD(learning_rate=0.01)

@nnx.jit # from flax.experimental import nnx
def train_step(model, x, y): # Model here is the Dense layer instance
    # Assuming the Dense layer instance passed IS the nnx.Module
    # This was a point of contention in previous tests.
    # The user's script implies the Keras layer (Dense) itself becomes the nnx.Module.
    
    # If the model passed is keras.Sequential, then model.layers[0] would be the nnx.Module
    # Let's assume for now the user intends to pass the Dense layer directly if this were
    # a more manual nnx model.
    # However, with Keras Sequential, we need to handle that the 'model' argument
    # might be the Sequential container.
    # For this test, let's assume the user's original intent was to test a single layer
    # that becomes an nnx.Module. Let's adapt if 'model' is Sequential.

    nnx_module_to_train = model # Default assumption
    if isinstance(model, keras.Sequential):
        if not model.built:
            # Build the model by calling it with sample data if it's Sequential and not built
            # Infer input shape from x or use a default if necessary
            sample_input_shape = x.shape
            if len(x.shape) == 1: # if x is a single sample (10,)
                 sample_input_shape = (1,) + x.shape # add batch dim (1,10)
            elif len(x.shape) == 2: # if x is (batch, features)
                 pass # shape is already good
            else:
                raise ValueError("Input x has unsupported shape for auto-build")
            model.build(input_shape=sample_input_shape)

        if not model.layers:
            raise ValueError("Sequential model has no layers.")
        nnx_module_to_train = model.layers[0] # Train the first layer
        print(f"Training the first layer of Sequential model: {nnx_module_to_train.name}")


    graphdef, params, state = nnx_module_to_train.split()

    def loss_fn_for_grad(current_params, current_state, x_batch, y_batch):
        # Apply graphdef with current_params and current_state
        output, new_state = graphdef.apply(current_params, current_state)(x_batch)
        loss = mse_loss(y_batch, output)
        return loss, new_state

    (loss, new_module_state), grads = nnx.value_and_grad(loss_fn_for_grad, argnums=0, has_aux=True)(params, state, x, y)
    
    # Access trainable_variables from the nnx_module_to_train (the Dense layer)
    if not hasattr(nnx_module_to_train, 'trainable_variables'):
        raise AttributeError("The module to train must have 'trainable_variables'.")

    flat_grads = jax.tree_util.tree_leaves(grads)
    
    if len(flat_grads) != len(nnx_module_to_train.trainable_variables):
        raise ValueError(
            f"Mismatch: {len(flat_grads)} gradients, "
            f"{len(nnx_module_to_train.trainable_variables)} trainable_variables."
        )

    optimizer.apply_gradients(zip(flat_grads, nnx_module_to_train.trainable_variables))

    # Update the specific layer (nnx_module_to_train)
    # This does not update the Sequential container directly.
    # For a real training loop with Sequential, one would need to reconstruct/update
    # the Sequential model or manage states of its layers.
    # For this test, we focus on whether the Dense layer's params change.
    # We can't directly call 'model.merge' if 'model' is Sequential.
    # Instead, we'd update the layer and then the Sequential would use the updated layer.
    # However, nnx.Module instances are typically what you merge.
    
    # The user's original script assumed 'model' is the nnx.Module to be split/merged.
    # If nnx_module_to_train is model.layers[0], we need a way to put it back or
    # ensure the Sequential model uses the updated layer's state.
    # This is a known complexity of mixing Keras Sequential with direct NNX state management.
    # For now, let's return the updated layer and check its weights.
    updated_layer_instance = nnx_module_to_train.merge(graphdef, params, new_module_state)
    
    return loss, updated_layer_instance # Return the specific layer that was trained


# --- End of user-provided @nnx.jit train_step ---

# Add a simple training loop and test execution
def main():
    print("Starting main test execution...")
    # Generate dummy data
    x_train = np.random.rand(100, 10).astype(np.float32)
    y_train = (np.sum(x_train * np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3, 0.4, -0.5]), axis=1) + 0.05).reshape(-1, 1).astype(np.float32)

    # Get initial weights of the Dense layer for comparison
    # Ensure model is built first (e.g. by calling it or model.build)
    # model.build(input_shape=(None, 10)) # or x_train.shape
    # The train_step will build it if it's Sequential and not built.

    dense_layer = model.layers[0]
    initial_weights = [np.array(w) for w in dense_layer.get_weights()]
    print(f"Initial weights of Dense layer: {initial_weights}")

    epochs = 5
    for epoch in range(epochs):
        loss, updated_dense_layer = train_step(model, x_train, y_train) # Pass Sequential model
        # After train_step, model.layers[0] is NOT automatically the updated_dense_layer.
        # We need to manually update it if we want to see changes reflected in the Sequential model.
        # For this test, we are checking if updated_dense_layer has changed weights.
        model.layers[0] = updated_dense_layer # Manually update the layer in Sequential
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    # Get final weights for comparison
    final_weights = [np.array(w) for w in updated_dense_layer.get_weights()]
    print(f"Final weights of Dense layer: {final_weights}")

    # Check if weights have changed
    weights_changed = False
    if len(initial_weights) == len(final_weights):
        for i_w, f_w in zip(initial_weights, final_weights):
            if not np.allclose(i_w, f_w):
                weights_changed = True
                break
    else:
        weights_changed = True # Different number of weight tensors means change

    if weights_changed:
        print("Weights changed after training!")
    else:
        print("Weights DID NOT change after training.")

    assert weights_changed, "Test failed: Model weights did not change after training."
    print("Test script completed successfully, weights changed.")

if __name__ == "__main__":
    main()

# --- End of test execution additions ---
