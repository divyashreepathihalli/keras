import keras
import numpy as np

np.random.seed(42)


model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=(10,), name="my_dense_layer") # Added a name for clarity
])

print("--- Initial Model Summary & Weights ---")
model.summary()

initial_weights = model.get_weights() # Gets a list of numpy arrays
print(f" Kernelweights{initial_weights[0]}")
print(f"Initial Bias:{initial_weights[1]}")

# dummy data
num_samples = 100
input_features = 10

# X: Random input data
X_dummy = np.random.rand(num_samples, input_features)

y_dummy = X_dummy[:, 0] + X_dummy[:, 1] * 0.5 + np.random.randn(num_samples) * 0.1
y_dummy = y_dummy.reshape(-1, 1)

# Build the model explicitly
model.build(input_shape=(None, 10)) # Add this line

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')                 


print("Train model")
history = model.fit(X_dummy, y_dummy, epochs=5, batch_size=32, verbose=1)

# Verify that variables are updated
print("--- Weights After Training ---")
updated_weights = model.get_weights()
print(f"Updated Kernel (weights):\n{updated_weights[0]}")
print(f"Updated Bias:\n{updated_weights[1]}")

# Compare initial and updated weights
variables_changed = False
print("\n--- Verification of Weight Updates ---")

# Compare Kernels
if not np.array_equal(initial_weights[0], updated_weights[0]):
    print("Kernel (weights) HAVE changed.")
    variables_changed = True
else:
    print("Kernel (weights) have NOT changed.")

# Compare Biases
if not np.array_equal(initial_weights[1], updated_weights[1]):
    print("Bias HAS changed.")
    variables_changed = True
else:
    print("Bias has NOT changed.")

if variables_changed:
    print("\nSUCCESS: At least one set of trainable variables was updated during training.")
else:
    print("\nFAILURE: No trainable variables seem to have been updated. Check training setup.")
