import jax.numpy as jnp
import jax
from flax import nnx
import os
os.environ["KERAS_BACKEND"]="jax"
os.environ["KERAS_NNX_ENABLED"]="true"
import keras
import numpy as np

print(f"Keras backend: {keras.backend.backend()}")
print(f"Keras NNX enabled: {keras.config.is_nnx_enabled()}")

# --- Test Config ---
RUN_MINIMAL_KERAS_LAYER_TEST = False # Known working
RUN_SEQUENTIAL_MODEL_TEST = False    # Known working, Pytree mismatch resolved by using nnx.Param for optimizer's wrt
RUN_FUNCTIONAL_MODEL_TEST = True
RUN_SUBCLASSED_MODEL_TEST = False

# --- Minimal Keras Dense Layer Test (Optional) ---
if RUN_MINIMAL_KERAS_LAYER_TEST:
    print("\n[NNX_DEBUG] Starting Minimal Keras Dense Layer Test")
    keras.config.set_dtype_policy("float32")
    # ... (Minimal test code - confirmed working, output can be verbose) ...
    print("[NNX_DEBUG] Finished Minimal Keras Dense Layer Test.")

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

# --- Functional Model Test ---
if RUN_FUNCTIONAL_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Functional Model Test...")
    keras.config.set_dtype_policy("float16")
    print(f"[NNX_DEBUG] Functional Test - Keras dtype policy: {keras.config.dtype_policy().name}")

    input_tensor = keras.Input(shape=(1,), name="input_1", dtype="float32")
    # Ensure layers within Functional model also have unique names if accessed that way
    dense_output = keras.layers.Dense(1, name="dense_functional_1")(input_tensor)
    functional_model = keras.Model(inputs=input_tensor, outputs=dense_output, name="functional_model_1")

    functional_tx = optax.sgd(1e-3)
    functional_optimizer_wrt = nnx.Param
    functional_optimizer = nnx.Optimizer(functional_model, functional_tx, wrt=functional_optimizer_wrt)

    @nnx.jit
    def functional_train_step(model_arg, optimizer_arg, batch):
        x, y = batch
        def loss_fn(model_for_loss):
            y_pred = model_for_loss(x)
            return jnp.mean((y - y_pred) ** 2)

        grads = nnx.grad(loss_fn, argnums=0)(model_arg)

        try:
            # For Functional models, children are named `_nnx_internal_op_0`, `_nnx_internal_op_1` etc.
            # op_0 is InputLayer, op_1 is Dense layer
            op1_grads = grads.get('_nnx_internal_op_1')
            if op1_grads:
                kernel_grads = op1_grads.get('_kernel')
                if kernel_grads and hasattr(kernel_grads, 'actual_param'):
                     jax.debug.print("kernel grad (Functional)-->{}", kernel_grads.actual_param.value)

                bias_grads = op1_grads.get('bias')
                if bias_grads and hasattr(bias_grads, 'actual_param'):
                    jax.debug.print("bias grad (Functional)-->{}", bias_grads.actual_param.value)
            else:
                # Fallback check if structure is via _layers (less likely now with explicit attrs)
                layers_in_grads = grads.get('_layers')
                if layers_in_grads and isinstance(layers_in_grads, dict) and 1 in layers_in_grads:
                    dense_grads_node = layers_in_grads[1]
                    # ... (further checks as in Sequential)
                    jax.debug.print("Functional grads found via _layers path. Full: {}", dense_grads_node)
                else:
                    jax.debug.print("Functional grads: Neither _nnx_internal_op_1 nor _layers[1] found. Full grads: {}", grads)
        except Exception as e:
            jax.debug.print("Error during debug print of Functional grads: {e_str}. Full grads: {grads_full}", e_str=str(e), grads_full=grads)

        optimizer_arg.update(grads)
        return model_arg, optimizer_arg

    current_fm_state = functional_model
    current_fm_optimizer_state = functional_optimizer

    print("\n[NNX_DEBUG] Debugging Pytree Structures for Functional Optimizer")
    fm_debug_batch = next(common_dataset())
    def _fm_debug_loss(m, x_arg, y_arg):
        y_pred = m(x_arg)
        return jnp.mean((y_pred - y_arg) ** 2)
    fm_debug_grads = nnx.grad(_fm_debug_loss, argnums=0)(current_fm_state, fm_debug_batch[0], fm_debug_batch[1])
    print(f"[NNX_DEBUG] Opti Debug - grads structure for Functional (raw type): {type(fm_debug_grads)}")
    # print(f"[NNX_DEBUG] Opti Debug - grads structure for Functional (str): {str(fm_debug_grads)}") # Potentially too verbose

    try:
        fm_params_state, _ = nnx.split(current_fm_optimizer_state.model, current_fm_optimizer_state.wrt)
        print(f"[NNX_DEBUG] Opti Debug - params_state for Functional (raw type from nnx.split): {type(fm_params_state)}")
        # print(f"[NNX_DEBUG] Opti Debug - params_state for Functional (str): {str(fm_params_state)}") # Potentially too verbose

        def get_safe_repr_type_only(x): # Simplified further
            return type(x)

        fm_param_types_repr = jax.tree.map(get_safe_repr_type_only, fm_params_state)
        print(f"[NNX_DEBUG] Opti Debug - param_TYPES for Functional: {fm_param_types_repr}")
        fm_grad_types_repr = jax.tree.map(get_safe_repr_type_only, fm_debug_grads)
        print(f"[NNX_DEBUG] Opti Debug - grad_TYPES for Functional: {fm_grad_types_repr}")

        # Compare full structures using jax.tree.structure
        print(f"[NNX_DEBUG] Opti Debug - Structure of params_state_for_optax (Functional): {jax.tree.structure(fm_params_state)}")
        print(f"[NNX_DEBUG] Opti Debug - Structure of grads_state (Functional): {jax.tree.structure(fm_debug_grads)}")


    except Exception as e:
        print(f"[NNX_DEBUG] Opti Debug - Error inspecting Functional optimizer params_state: {e}")
        import traceback; traceback.print_exc()
    print("[NNX_DEBUG] End Debugging Pytree Structures (Functional Model)\n")

    for step, batch_data in enumerate(common_dataset()):
        current_fm_state, current_fm_optimizer_state = functional_train_step(current_fm_state, current_fm_optimizer_state, batch_data)
        if step % 100 == 0:
            logs = common_test_step(current_fm_state, (X_np, Y_np))
            print(f"Functional step: {step}, loss: {logs['loss']}")
        if step >= 1000: break # Reduced steps for faster test cycle
    print("Keras Functional Test Run finished.")

    print("Final Keras Functional weights:")
    try:
        final_dense_layer = current_fm_state.get_layer("dense_functional_1") # Name used in Functional def
        if final_dense_layer and hasattr(final_dense_layer, 'kernel') and hasattr(final_dense_layer.kernel, 'actual_param'):
            print(f"  Kernel: {final_dense_layer.kernel.actual_param.value}")
        if final_dense_layer and hasattr(final_dense_layer, 'bias') and hasattr(final_dense_layer.bias, 'actual_param'):
            print(f"  Bias: {final_dense_layer.bias.actual_param.value}")
    except Exception as e:
        print(f"  Error printing Functional final weights: {e}")

    fm_final_grads = nnx.grad(_fm_debug_loss, argnums=0)(current_fm_state, fm_debug_batch[0], fm_debug_batch[1])
    print("Final Keras Functional gradients:")
    # print(f"  Full final_grads structure: {fm_final_grads}") # Verbose
    # Refined grad printing for functional
    op1_final_grads = fm_final_grads.get('_nnx_internal_op_1')
    if op1_final_grads:
        kernel_final_grads = op1_final_grads.get('_kernel')
        if kernel_final_grads and hasattr(kernel_final_grads, 'actual_param'):
            print(f"  Kernel grad: {kernel_final_grads.actual_param.value if hasattr(kernel_final_grads.actual_param, 'value') else kernel_final_grads.actual_param}")
        bias_final_grads = op1_final_grads.get('bias')
        if bias_final_grads and hasattr(bias_final_grads, 'actual_param'):
            print(f"  Bias grad: {bias_final_grads.actual_param.value if hasattr(bias_final_grads.actual_param, 'value') else bias_final_grads.actual_param}")
    else:
        print("  Final grads for Functional model don't have _nnx_internal_op_1 path.")


# --- Keras Sequential Model Test (Optional - if RUN_SEQUENTIAL_MODEL_TEST is True) ---
if RUN_SEQUENTIAL_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Sequential Test...")
    # ... (Sequential model test code as before, known to work now) ...
    print("Sequential model test was skipped by config.")


# --- Subclassed Model Test (Optional - if RUN_SUBCLASSED_MODEL_TEST is True) ---
if RUN_SUBCLASSED_MODEL_TEST:
    print("\n[NNX_DEBUG] Starting Keras Subclassed Model Test...")
    keras.config.set_dtype_policy("float16")
    print(f"[NNX_DEBUG] Subclassed Test - Keras dtype policy: {keras.config.dtype_policy().name}")

    class MySimpleModel(keras.Model):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.dense_layer = keras.layers.Dense(1, name="dense_subclassed_1")
       def call(self, inputs):
           return self.dense_layer(inputs)

    subclassed_model = MySimpleModel(name="subclassed_model_1")
    subclassed_model.build(X_np.shape)

    subclassed_tx = optax.sgd(1e-3)
    subclassed_optimizer_wrt = nnx.Param
    subclassed_optimizer = nnx.Optimizer(subclassed_model, subclassed_tx, wrt=subclassed_optimizer_wrt)

    @nnx.jit
    def subclassed_train_step(model_arg, optimizer_arg, batch):
        x, y = batch
        def loss_fn(model_for_loss):
            y_pred = model_for_loss(x)
            return jnp.mean((y - y_pred) ** 2)
        grads = nnx.grad(loss_fn, argnums=0)(model_arg)
        try:
            # For subclassed model, grads should have 'dense_layer' attribute
            dense_grads = grads.get('dense_layer')
            if dense_grads:
                if hasattr(dense_grads, '_kernel') and hasattr(dense_grads._kernel, 'actual_param'):
                     jax.debug.print("kernel grad (Subclassed)-->{}", dense_grads._kernel.actual_param.value)
                if hasattr(dense_grads, 'bias') and hasattr(dense_grads.bias, 'actual_param'):
                     jax.debug.print("bias grad (Subclassed)-->{}", dense_grads.bias.actual_param.value)
            else:
                jax.debug.print("Subclassed grads: dense_layer not found. Full grads: {}", grads)
        except Exception as e:
            jax.debug.print("Error during debug print of Subclassed grads: {e_str}", e_str=str(e))
        optimizer_arg.update(grads)
        return model_arg, optimizer_arg

    current_sc_state = subclassed_model
    current_sc_optimizer_state = subclassed_optimizer

    print("\n[NNX_DEBUG] Debugging Pytree Structures for Subclassed Optimizer")
    sc_debug_batch = next(common_dataset())
    def _sc_debug_loss(m, x_arg, y_arg):
        y_pred = m(x_arg)
        return jnp.mean((y_pred - y_arg) ** 2)
    sc_debug_grads = nnx.grad(_sc_debug_loss, argnums=0)(current_sc_state, sc_debug_batch[0], sc_debug_batch[1])
    print(f"[NNX_DEBUG] Opti Debug - grads structure for Subclassed (str): {str(sc_debug_grads)}")
    try:
        sc_params_state, _ = nnx.split(current_sc_optimizer_state.model, current_sc_optimizer_state.wrt)
        print(f"[NNX_DEBUG] Opti Debug - params_state for Subclassed (str): {str(sc_params_state)}")
        print(f"[NNX_DEBUG] Opti Debug - Structure of params_state (Subclassed): {jax.tree.structure(sc_params_state)}")
        print(f"[NNX_DEBUG] Opti Debug - Structure of grads_state (Subclassed): {jax.tree.structure(sc_debug_grads)}")
    except Exception as e:
        print(f"[NNX_DEBUG] Opti Debug - Error inspecting Subclassed optimizer params_state: {e}")
    print("[NNX_DEBUG] End Debugging Pytree Structures (Subclassed Model)\n")


    for step, batch_data in enumerate(common_dataset()):
        current_sc_state, current_sc_optimizer_state = subclassed_train_step(current_sc_state, current_sc_optimizer_state, batch_data)
        if step % 100 == 0:
            logs = common_test_step(current_sc_state, (X_np, Y_np))
            print(f"Subclassed step: {step}, loss: {logs['loss']}")
        if step >= 1000: break
    print("Keras Subclassed Test Run finished.")
    # ... (final weight and grad prints for subclassed model) ...


print("Full test script finished.")
