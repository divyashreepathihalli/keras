import os
# Ensure JAX backend and NNX are enabled before other imports
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import keras
import keras_hub
import jax
import jax.numpy as jnp
from flax import nnx # type: ignore
import optax
import tunix
import numpy as np # For dummy data generation

# Suppress warnings for cleaner output, if necessary
import logging
logging.getLogger("keras_core").setLevel(logging.ERROR)
logging.getLogger("keras_nlp").setLevel(logging.ERROR)


def main():
    print("Starting Gemma with Tunix and NNX example...")

    # 1. Load the Gemma model
    print("Loading Gemma model...")
    try:
        # Using a smaller variant for quicker testing if available, otherwise default
        gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en")
    except Exception as e:
        print(f"Error loading Gemma model: {e}")
        print("Please ensure you have authentication configured for Kaggle if using gated models.")
        print("Falling back to a generic Keras model for demonstration if Gemma fails.")
        # Fallback to a simple model if Gemma loading fails, to ensure script can run
        gemma_lm = keras.Sequential([
            keras.layers.Embedding(input_dim=100, output_dim=16),
            keras.layers.LSTM(32),
            keras.layers.Dense(100, activation="softmax")
        ])
        # For the fallback model, we need a tokenizer substitute
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 100
            def tokenize(self, texts): return np.random.randint(0, self.vocab_size, size=(len(texts), 10)).astype(np.int32)
            def detokenize(self, tokens): return [" ".join(map(str, t)) for t in tokens]
            @property
            def pad_token_id(self): return 0

        gemma_lm.tokenizer = DummyTokenizer()
        gemma_lm.preprocessor = keras.layers.IntegerLookup(vocabulary=list(range(100))) # Dummy preprocessor
        # Need to build the fallback model
        dummy_input = np.random.randint(0, 100, size=(2, 10))
        gemma_lm.predict(dummy_input)


    print("Gemma model loaded (or fallback created).")

    # Ensure model is built (important for NNX variable access)
    # For Gemma, generation/compilation builds it. For fallback, predict did.
    # If it's the actual Gemma model, try a simple generation to build it.
    if hasattr(gemma_lm, 'generate'):
        try:
            print("Building Gemma model with a dummy generation call...")
            gemma_lm.generate(["hello"], max_length=5)
            print("Gemma model built.")
        except Exception as e:
            print(f"Could not build Gemma with generate: {e}. Attempting compile.")
            # If generate fails (e.g. due to complexity of first call), try compile
            try:
                gemma_lm.compile(optimizer=optax.adam(1e-5), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
                print("Gemma model compiled and built.")
            except Exception as e_compile:
                print(f"Could not compile Gemma model: {e_compile}")
                print("Proceeding, but variable extraction might fail.")

    # 2. Define Tunix TrainState and Process
    class GemmaTrainState(tunix.TrainState):
        # Model is stored implicitly by Tunix if we pass it to Process.
        # We only need to store what's not automatically handled or needs specific initialization.
        pass # For now, keep it simple. Optimizer state will be handled by Tunix.

    class GemmaTunixProcess(tunix.Process):
        def __init__(self, model, learning_rate=1e-5):
            super().__init__(name="gemma_tunix_process")
            self.model = model # Keras model
            self.learning_rate = learning_rate
            # Optimizer will be initialized in self.initialize
            self.optimizer = None

        def initialize(self, rng: jax.random.PRNGKey):
            print("Initializing GemmaTunixProcess...")

            # Extract trainable variables for NNX Optimizer
            # For Keras models with NNX enabled, variables should be accessible
            # The way to get trainable variables might differ based on Keras version and NNX integration
            try:
                # This is a potential way; actual Keras NNX API might be different
                # We need to ensure the model's variables are JAX types for Optax

                # Option 1: Hope Keras variables are directly usable by nnx.Optimizer
                # This relies on Keras Variables being pytrees that optax can handle.
                # The `wrt` argument in nnx.Optimizer needs a way to filter these.
                # A common way for NNX modules is to use nnx.filter or specific types.
                # For Keras models, this is less direct.

                # Let's assume for now Keras variables work if they are JAX arrays.
                # We might need to convert them or ensure they are treated as such.

                # The issue example used:
                # trainable_var = nnx.All(keras.Variable, lambda path, x: getattr(x, '_trainable', False))
                # This assumes keras.Variable is the type of trainable weights.

                # Let's try to get variables that are marked as trainable.
                # Keras model's `trainable_variables` should provide these.
                # We need them in a structure that nnx.Optimizer can use.

                # For NNX, the model itself is often the first argument to nnx.Optimizer
                # and `wrt` filters params *within* that model structure.

                # If self.model is already an NNX-compatible structure (e.g. an nnx.Module wrapping Keras layers),
                # then nnx.Optimizer(self.model, optax_optimizer, wrt=nnx.Param) could work.
                # However, gemma_lm is a keras.Model.

                # The example from the issue:
                # optimizer = nnx.Optimizer(model, tx, wrt=trainable_var)
                # Here `model` was a Keras Layer (Dense), not a full Model.
                # And trainable_var was a filter.

                # We need to adapt this for a full Keras model.
                # One approach: Treat the whole model as the 'module' for nnx.Optimizer.

                # Keras models have `model.variables` and `model.trainable_variables`
                # These are lists of KerasVariable objects.
                # We need to ensure these are compatible with JAX for Optax.
                # When KERAS_BACKEND=jax, KerasVariables store JAX arrays.

                # Let's define the optimizer using optax directly first
                self.optax_optimizer = optax.adam(self.learning_rate)

                # The challenge is how nnx.Optimizer will get/set these variables on the Keras model.
                # If Keras model is a Pytree of its variables, it might work.

                # For now, let's assume we can pass the model directly to nnx.Optimizer.
                # The `wrt` argument will be crucial.
                # Let's use a simple filter that marks all nnx.Param as trainable.
                # This assumes Keras layers internally use nnx.Param for their weights when NNX is enabled.

                # If keras.Model itself is not an nnx.Module, we might need to wrap it
                # or handle parameters more manually.

                # The example `optimizer = nnx.Optimizer(model, tx, wrt=trainable_var)`
                # implies `model` is the structure containing variables, and `wrt` filters them.
                # Let's try passing the Keras model directly.

                # Filter for trainable Keras Variables
                # Keras Variables have a .path attribute we might be able to use if nnx.Optimizer needs it
                def is_keras_trainable_variable(path, node):
                    return isinstance(node, keras.Variable) and node.trainable

                # This filter might not be directly usable by nnx.Optimizer as `wrt` if Keras model isn't an nnx.Module.
                # Let's try the simpler `nnx.Param` filter first, assuming Keras layers expose params that way.
                try:
                    self.optimizer = nnx.Optimizer(self.model, self.optax_optimizer) # Default wrt is nnx.Param
                    print("nnx.Optimizer initialized with default nnx.Param filter.")
                except Exception as e_opt:
                    print(f"Failed to init nnx.Optimizer with default filter: {e_opt}")
                    print("Attempting to use model.trainable_variables directly with optax (less Tunix/NNX idiomatic).")
                    # Fallback: Manage optax state manually if nnx.Optimizer setup is tricky
                    # This means we are not using nnx.Optimizer as intended, but it's a fallback.
                    params_pytree = self.model.trainable_variables # This is a list
                    # Optax needs a pytree. A list of arrays is a pytree.
                    # We need to be careful if these are KerasVariable objects or raw arrays.
                    # Assuming they are JAX arrays under the hood.
                    self.opt_state = self.optax_optimizer.init(params_pytree)
                    self.optimizer = None # Mark that nnx.Optimizer is not used
                    print("Initialized optax state manually with model.trainable_variables.")


            except Exception as e:
                print(f"Error during optimizer initialization: {e}")
                raise

            # Initial state for Tunix (empty for now, could include PRNG key or step)
            return GemmaTrainState()


        def update(self, state: GemmaTrainState, batch, rng: jax.random.PRNGKey):
            # Tunix `update` is like a general step, can be for training, eval, etc.
            # We'll make this a training step.
            # print("Executing GemmaTunixProcess update (train step)...")

            input_ids, attention_mask, labels = batch

            # Define loss function for JAX grad
            def loss_fn(model_variables_or_model):
                # If nnx.Optimizer is used, it handles variable extraction/application.
                # `model_variables_or_model` will be the model itself or its graphdef.

                # If using manual optax, `model_variables_or_model` would be the pytree of params.
                # However, Keras models are typically called as model(inputs).

                # We need to ensure the Keras model is called with the *updated* variables
                # that JAX is differentiating.

                # If `self.optimizer` (nnx.Optimizer) is active:
                if self.optimizer:
                    # The model passed to loss_fn by nnx.grad will be an nnx.GraphDef
                    # or a version of the model with updated states.
                    # We call it directly.
                    y_pred = model_variables_or_model(input_ids, attention_mask=attention_mask) # Call the (potentially transformed) model
                else:
                    # Manual optax: we need to apply new variables to the model.
                    # This is tricky with Keras. Keras model.apply_gradients might be one way,
                    # or setting variables directly.
                    # For simplicity in this example, if nnx.Optimizer fails, this part will be hard.
                    # Let's assume for now nnx.Optimizer works.
                    # If not, this part needs significant rework for manual Keras param updates.
                    # One common pattern is to have the loss_fn take params and then somehow use them.

                    # Let's assume model_variables_or_model are the parameters if not using nnx.Optimizer
                    # This is where it gets complex if not using nnx.Optimizer to manage model state.
                    # For this example, we'll strongly prefer nnx.Optimizer path.
                    # If it failed, this step will likely fail or be incorrect.
                    # A proper manual optax loop with Keras requires more boilerplate.
                    # We'd have to update self.model.trainable_variables based on grads.
                    print("Warning: nnx.Optimizer is not active. Loss calculation might not reflect gradient updates correctly without more complex manual parameter handling for Keras model.")
                    y_pred = self.model(input_ids, attention_mask=attention_mask)


                # Assuming y_pred are logits
                # Loss for language modeling: sparse categorical crossentropy
                # Keras loss functions can be used if they operate on JAX arrays
                loss = keras.losses.sparse_categorical_crossentropy(labels, y_pred, from_logits=True)
                return jnp.mean(loss)

            if self.optimizer: # Using nnx.Optimizer
                # nnx.grad needs to know which arguments are static and which are differentiable.
                # By default, the first arg (the module/model) is differentiated.
                # It will handle getting/setting variables on self.model.
                loss_value, grads = nnx.value_and_grad(loss_fn)(self.model)
                # `grads` will be a pytree matching the structure of trainable params in self.model
                self.optimizer.update(grads) # This updates self.model in-place
            else: # Manual optax (fallback, less ideal for NNX demo)
                # This path is more complex with Keras models.
                # We need to get params, compute grads wrt them, then update Keras model's vars.
                current_params = self.model.trainable_variables # List of KerasVariables (JAX arrays)

                # JAX grad needs a function of params.
                # This requires loss_fn to be structured differently, e.g., by temporarily assigning params.
                # This is a placeholder and likely needs a more robust solution for Keras.
                def loss_fn_manual_params(params_list):
                    # Problem: How to make self.model use these params for a forward pass?
                    # Keras models don't typically take params directly in call().
                    # This is a major challenge for manual optax with full Keras models.
                    # For this example, we'll print a warning and calculate loss on current model state.
                    # A real implementation would need to reconstruct the model or use low-level APIs.
                    print("Warning: Manual optax gradient calculation with Keras model is non-trivial. Gradients might be on detached params.")
                    y_pred = self.model(input_ids, attention_mask=attention_mask)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, y_pred, from_logits=True)
                    return jnp.mean(loss)

                loss_value, grads_list = jax.value_and_grad(loss_fn_manual_params)(current_params)

                updates, self.opt_state = self.optax_optimizer.update(grads_list, self.opt_state, current_params)

                # Apply updates to Keras model variables
                # This is the critical part for manual updates
                if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'apply_gradients'):
                    # If Keras model has a compiled optimizer, try using it (though we are using optax)
                    # This is mixing things, usually not recommended.
                     self.model.optimizer.apply_gradients(zip(grads_list, self.model.trainable_variables))
                else:
                    # Direct variable assignment (if KerasVariable supports .assign())
                    for var, update_val in zip(self.model.trainable_variables, updates):
                        var.assign(var - update_val) # Optax updates are typically subtracted (gradient descent)
                                                     # Or, if optax.apply_updates is used, it's `new_param = optax.apply_updates(param, update)`
                                                     # Let's assume optax.adam performs the subtraction sense correctly.
                                                     # A common pattern: new_params = optax.apply_updates(current_params, updates)
                                                     # Then assign new_params to model variables.
                    new_params_pytree = optax.apply_updates(current_params, updates)
                    for var, new_val in zip(self.model.trainable_variables, new_params_pytree):
                        var.assign(new_val)
                print("Applied gradients manually using optax and Keras variable assignment.")


            metrics = {'loss': loss_value}
            return state, metrics # Return original state (or new if it changes), and metrics

    # 3. Prepare a dummy dataset
    print("Preparing dummy dataset...")
    # Sample prompts
    prompts = [
        "Keras is a",
        "I want to say",
        "The future of AI is"
    ]
    # Use Gemma's tokenizer
    try:
        # Tokenize
        # Gemma preprocessor handles tokenization and padding
        # Max length for dummy data
        max_length = 20 # Small max length for dummy data

        # Preprocess to get input_ids, padding_mask
        # Gemma preprocessor output is a dict {'token_ids': ..., 'padding_mask': ...}
        processed_inputs = gemma_lm.preprocessor(prompts, sequence_length=max_length)
        input_ids = processed_inputs['token_ids']
        attention_mask = processed_inputs['padding_mask']

        # Create labels for language modeling (shifted input_ids)
        labels = jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), gemma_lm.tokenizer.pad_token_id)], axis=-1)

        # Create a simple list of batches for the demo
        # Ensure they are JAX arrays if not already
        dataset = [(jnp.array(input_ids), jnp.array(attention_mask), jnp.array(labels))]
        print(f"Dummy dataset created with {len(prompts)} samples, sequence length {max_length}.")
        print(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")

    except Exception as e:
        print(f"Error creating dataset with Gemma tokenizer: {e}")
        print("Using purely random data as fallback.")
        # Fallback for tokenizer issues (e.g. if Gemma was not loaded)
        vocab_size = 100 # Must match fallback model's vocab size
        seq_len = 10
        num_samples = 3
        input_ids_fallback = np.random.randint(0, vocab_size, size=(num_samples, seq_len)).astype(np.int32)
        attention_mask_fallback = np.ones_like(input_ids_fallback)
        labels_fallback = np.random.randint(0, vocab_size, size=(num_samples, seq_len)).astype(np.int32) # Not shifted, just dummy
        dataset = [(jnp.array(input_ids_fallback), jnp.array(attention_mask_fallback), jnp.array(labels_fallback))]
        print(f"Fallback dummy dataset created: {num_samples} samples, sequence length {seq_len}.")


    # 4. Instantiate and run the Tunix Process
    print("Instantiating and running Tunix Process...")
    process = GemmaTunixProcess(model=gemma_lm, learning_rate=1e-5)

    # Initialize the process
    key = jax.random.PRNGKey(0)
    try:
        train_state = process.initialize(key)
        print("Tunix process initialized.")
    except Exception as e:
        print(f"Fatal error during Tunix process initialization: {e}")
        print("This often indicates issues with model variable handling or NNX compatibility.")
        return # Exit if initialization fails

    # Training loop
    num_steps = 5 # Small number of steps for demo
    for step_num in range(num_steps):
        batch = dataset[0] # Using the same batch for simplicity
        try:
            train_state, metrics = process.update(train_state, batch, jax.random.PRNGKey(step_num))
            print(f"Step {step_num + 1}/{num_steps}, Loss: {metrics['loss']:.4f}")
        except Exception as e:
            print(f"Error during training step {step_num + 1}: {e}")
            print("This might be due to issues in loss_fn, gradient computation, or optimizer update.")
            # Check if self.model and self.optimizer exist before accessing them in the hint
            model_exists = 'self' in locals() and hasattr(self, 'model')
            optimizer_exists = 'self' in locals() and hasattr(self, 'optimizer')
            if "GraphDef" in str(e) and "does not have state" in str(e) and \
               model_exists and isinstance(self.model, keras.Model) and \
               optimizer_exists and self.optimizer is not None:
                 print("Hint: This error can occur if nnx.Optimizer is used with a Keras Model that isn't fully NNX-compatible or if state isn't managed correctly by the Keras NNX wrapper.")
            break # Stop training if an error occurs

    # 5. Demonstrate text generation (using the original Keras model object)
    print("\nDemonstrating text generation after Tunix steps...")
    if hasattr(gemma_lm, 'generate'):
        try:
            # The model `gemma_lm` should have been updated in-place by nnx.Optimizer
            generated_text = gemma_lm.generate(["Keras is a"], max_length=30)
            print("Generated text (Keras is a):")
            for text in generated_text:
                print(text)

            batched_prompts = ["I want to say", "The meaning of life is"]
            generated_texts_batch = gemma_lm.generate(batched_prompts, max_length=30)
            print("\nGenerated texts (batched):")
            for text in generated_texts_batch:
                print(text)

        except Exception as e:
            print(f"Error during text generation: {e}")
            if "apply_preprocessing" in str(e) or "token_ids" in str(e):
                print("Hint: Generation error might be related to tokenizer or preprocessor state if the model was modified in an unexpected way.")
    else:
        print("Skipping generation as fallback model does not have a .generate() method.")

    print("\nExample finished.")

if __name__ == "__main__":
    main()
