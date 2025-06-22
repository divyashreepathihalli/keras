import jax.numpy as jnp
import jax
from flax import nnx
import os
os.environ["KERAS_BACKEND"]="jax"
os.environ["KERAS_NNX_ENABLED"]="true"
import keras
import numpy as np
import optax # <--- MOVED IMPORT TO GLOBAL SCOPE

print(f"Keras backend: {keras.backend.backend()}")
print(f"Keras NNX enabled: {keras.config.is_nnx_enabled()}")

# --- Common Setup for Model Tests ---
X_np = np.linspace(-jnp.pi, jnp.pi, 100)[:, None].astype(np.float32)
Y_np = (0.8 * X_np + 0.1 + np.random.normal(0, 0.1, size=X_np.shape)).astype(np.float32)

def common_dataset(batch_size=10):
 while True:
   idx = np.random.choice(len(X_np), size=batch_size)
   yield X_np[idx], Y_np[idx]

@nnx.jit
def common_test_step(model_arg, batch):
 x, y = batch
 y_pred = model_arg(x)
 loss = jnp.mean((y - y_pred) ** 2)
 return {'loss': loss}

# --- Test Config ---
RUN_MINIMAL_KERAS_LAYER_TEST = True
RUN_SEQUENTIAL_MODEL_TEST = True
RUN_FUNCTIONAL_MODEL_TEST = False
RUN_SUBCLASSED_MODEL_TEST = False

# --- Minimal Keras Dense Layer Test ---
if RUN_MINIMAL_KERAS_LAYER_TEST:
    print("\n[NNX_DEBUG] Starting Minimal Keras Dense Layer Test")
    keras.config.set_dtype_policy("float32")
    print(f"[NNX_DEBUG] Minimal Test - Keras dtype policy: {keras.config.dtype_policy().name}")

    dense_layer_minimal = keras.layers.Dense(1, name="minimal_dense")
    print(f"[NNX_DEBUG] Minimal Test - Dense layer created: {dense_layer_minimal.name}")
    input_shape_minimal = (3, 2)
    dense_layer_minimal.build(input_shape_minimal)
    print(f"[NNX_DEBUG] Minimal Test - Dense layer built. Path: {dense_layer_minimal.path}")
    if hasattr(dense_layer_minimal, '_kernel') and dense_layer_minimal._kernel is not None:
        print(f"[NNX_DEBUG] Minimal Test - dense_layer_minimal._kernel type: {type(dense_layer_minimal._kernel)}")
        if hasattr(dense_layer_minimal._kernel, 'actual_param'):
            print(f"[NNX_DEBUG] Minimal Test - dense_layer_minimal._kernel.actual_param type: {type(dense_layer_minimal._kernel.actual_param)}")
    if hasattr(dense_layer_minimal, 'bias') and dense_layer_minimal.bias is not None:
        print(f"[NNX_DEBUG] Minimal Test - dense_layer_minimal.bias type: {type(dense_layer_minimal.bias)}")
        if hasattr(dense_layer_minimal.bias, 'actual_param'):
            print(f"[NNX_DEBUG] Minimal Test - dense_layer_minimal.bias.actual_param type: {type(dense_layer_minimal.bias.actual_param)}")

    x_data_minimal = jnp.ones(input_shape_minimal, dtype=jnp.float32)
    y_data_minimal = jnp.ones((input_shape_minimal[0], 1), dtype=jnp.float32)

    def minimal_dense_loss_fn(layer_instance, x, y_true):
        y_pred = layer_instance(x)
        loss = jnp.mean((y_pred - y_true) ** 2)
        return loss

    try:
        grads_minimal_dense = nnx.grad(minimal_dense_loss_fn, argnums=0)(dense_layer_minimal, x_data_minimal, y_data_minimal)
        print(f"[NNX_DEBUG] Minimal Test - Grads calculated. Full grads_minimal_dense structure (type): {type(grads_minimal_dense)}")

        kernel_grad_attr_name_to_check = '_kernel'
        bias_grad_attr_name_to_check = 'bias'

        if hasattr(grads_minimal_dense, kernel_grad_attr_name_to_check) and getattr(grads_minimal_dense, kernel_grad_attr_name_to_check) is not None:
            kernel_grad_module_state = getattr(grads_minimal_dense, kernel_grad_attr_name_to_check)
            if hasattr(kernel_grad_module_state, 'actual_param') and kernel_grad_module_state.actual_param is not None:
                kernel_grad_val = kernel_grad_module_state.actual_param
                if hasattr(kernel_grad_val, 'value'): kernel_grad_val = kernel_grad_val.value
                print(f"[NNX_DEBUG] Minimal Test - Kernel grad value (via {kernel_grad_attr_name_to_check}.actual_param): shape {kernel_grad_val.shape}")
        else: print(f"[NNX_DEBUG] Minimal Test - Kernel grad (via {kernel_grad_attr_name_to_check}.actual_param) not found.")

        if hasattr(grads_minimal_dense, bias_grad_attr_name_to_check) and getattr(grads_minimal_dense, bias_grad_attr_name_to_check) is not None:
            bias_grad_module_state = getattr(grads_minimal_dense, bias_grad_attr_name_to_check)
            if hasattr(bias_grad_module_state, 'actual_param') and bias_grad_module_state.actual_param is not None:
                bias_grad_val = bias_grad_module_state.actual_param
                if hasattr(bias_grad_val, 'value'): bias_grad_val = bias_grad_val.value
                print(f"[NNX_DEBUG] Minimal Test - Bias grad value (via {bias_grad_attr_name_to_check}.actual_param): shape {bias_grad_val.shape}")
        elif hasattr(grads_minimal_dense, '_trainable_variables') and isinstance(grads_minimal_dense._trainable_variables, dict) and \
           1 in grads_minimal_dense._trainable_variables and grads_minimal_dense._trainable_variables[1] is not None and \
           hasattr(grads_minimal_dense._trainable_variables[1], 'actual_param'):
            bias_grad_alt_val = grads_minimal_dense._trainable_variables[1].actual_param
            if hasattr(bias_grad_alt_val, 'value'): bias_grad_alt_val = bias_grad_alt_val.value
            print(f"[NNX_DEBUG] Minimal Test - Bias grad value (via _trainable_variables[1].actual_param): shape {bias_grad_alt_val.shape}")
        else:
            print(f"[NNX_DEBUG] Minimal Test - Bias grad (via {bias_grad_attr_name_to_check}.actual_param or _trainable_variables[1]) not found.")
    except Exception as e:
        print(f"[NNX_DEBUG] Minimal Test - Error during gradient calculation or inspection: {e}")
        import traceback
        traceback.print_exc()
    print("[NNX_DEBUG] Finished Minimal Keras Dense Layer Test.")


# --- Keras Sequential Model Test ---
if RUN_SEQUENTIAL_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Sequential Test...")
    keras.config.set_dtype_policy("float16")
    print(f"[NNX_DEBUG] Sequential Test - Keras dtype policy: {keras.config.dtype_policy().name}")

    seq_model = keras.Sequential([keras.layers.Dense(1, name="dense_layer_in_seq")])
    seq_model.build(X_np.shape)

    seq_tx = optax.sgd(1e-3)
    seq_optimizer_wrt = nnx.Param
    seq_optimizer = nnx.Optimizer(seq_model, seq_tx, wrt=seq_optimizer_wrt)

    @nnx.jit
    def seq_train_step(model_arg, optimizer_arg, batch):
        x_batch, y_batch = batch
        def loss_fn(model_for_loss):
            y_pred = model_for_loss(x_batch)
            return jnp.mean((y_pred - y_batch) ** 2)

        grads = nnx.grad(loss_fn, argnums=0)(model_arg)

        try:
            if hasattr(grads, '_functional') and grads._functional is not None:
                jax.debug.print("Seq train_step: Grads has _functional key.")
            else:
                jax.debug.print("Seq train_step: Grads does NOT have _functional key. Full grads: {}", grads)
        except Exception as e:
            jax.debug.print("Error during simplified debug print of grads (Seq): {e_str}. Full grads: {grads_full}", e_str=str(e), grads_full=grads)

        optimizer_arg.update(grads)
        return model_arg, optimizer_arg

    current_seq_model_state = seq_model
    current_seq_optimizer_state = seq_optimizer

    print("\n[NNX_DEBUG] Debugging Pytree Structures for Sequential Optimizer")
    seq_debug_batch = next(common_dataset())
    def _seq_debug_loss(m, x, y):
        y_pred = m(x)
        return jnp.mean((y_pred - y) ** 2)

    seq_debug_grads_state = nnx.grad(_seq_debug_loss, argnums=0)(current_seq_model_state, seq_debug_batch[0], seq_debug_batch[1])
    print(f"[NNX_DEBUG] Opti Debug - grads_state for Sequential (type): {type(seq_debug_grads_state)}")
    print(f"[NNX_DEBUG] Opti Debug - grads_state for Sequential (structure): {jax.tree.structure(seq_debug_grads_state)}")

    try:
        params_state_for_optax, _ = nnx.split(current_seq_optimizer_state.model, current_seq_optimizer_state.wrt)
        print(f"[NNX_DEBUG] Opti Debug - params_state for Sequential (type from nnx.split): {type(params_state_for_optax)}")
        print(f"[NNX_DEBUG] Opti Debug - params_state for Sequential (structure from nnx.split): {jax.tree.structure(params_state_for_optax)}")

        def get_leaf_value_for_optax(x):
            return x.value if hasattr(x, 'value') else x

        seq_param_values = jax.tree.map(get_leaf_value_for_optax, params_state_for_optax)
        seq_grad_values = jax.tree.map(get_leaf_value_for_optax, seq_debug_grads_state)

        print(f"[NNX_DEBUG] Opti Debug - Structure of param_VALUES for Optax (Sequential): {jax.tree.structure(seq_param_values)}")
        print(f"[NNX_DEBUG] Opti Debug - Structure of grad_VALUES for Optax (Sequential): {jax.tree.structure(seq_grad_values)}")

    except Exception as e:
        print(f"[NNX_DEBUG] Opti Debug - Error inspecting Sequential optimizer Pytree structures: {e}")
        import traceback
        traceback.print_exc()
    print("[NNX_DEBUG] End Debugging Pytree Structures (Sequential Model)\n")

    for step, batch_data in enumerate(common_dataset()):
        current_seq_model_state, current_seq_optimizer_state = seq_train_step(current_seq_model_state, current_seq_optimizer_state, batch_data)
        if step % 100 == 0:
            logs = common_test_step(current_seq_model_state, (X_np, Y_np))
            print(f"Sequential step: {step}, loss: {logs['loss']}")
        if step >= 1000: break
    print("Keras Sequential Test Run finished.")

    print("Final Keras Sequential weights:")
    try:
        if current_seq_model_state._functional and hasattr(current_seq_model_state._functional, '_operations') and \
           len(current_seq_model_state._functional._operations) > 1:
            dense_layer_in_seq = current_seq_model_state._functional._operations[1]
            if hasattr(dense_layer_in_seq, 'kernel') and hasattr(dense_layer_in_seq.kernel, 'actual_param') and \
               hasattr(dense_layer_in_seq, 'bias') and hasattr(dense_layer_in_seq.bias, 'actual_param'):
                print(f"  Kernel: {dense_layer_in_seq.kernel.actual_param.value}")
                print(f"  Bias: {dense_layer_in_seq.bias.actual_param.value}")
    except Exception as e: print(f"  Error printing final Seq weights: {e}")
    # ... (final grad printing for sequential)


# --- Functional Model Test ---
if RUN_FUNCTIONAL_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Functional Model Test...")
    # ... (Functional model test code as before) ...
    print("Functional model test currently disabled by RUN_FUNCTIONAL_MODEL_TEST flag.")


# --- Subclassed Model Test ---
if RUN_SUBCLASSED_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Subclassed Model Test...")
    # ... (Subclassed model test code) ...
    print("Subclassed model test currently disabled by RUN_SUBCLASSED_MODEL_TEST flag.")

print("Full test script finished.")
