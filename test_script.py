import jax.numpy as jnp
import jax
from flax import nnx
import os
os.environ["KERAS_BACKEND"]="jax"
os.environ["KERAS_NNX_ENABLED"]="true"
import keras
keras.config.disable_traceback_filtering()
import numpy as np
import keras
from flax import nnx
import optax


X = np.linspace(-jnp.pi, jnp.pi, 100)[:, None]
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)


# Define the input layer
inputs = keras.Input(shape=(1,)) # Assuming a single input feature, adjust 'shape' as needed
x = keras.layers.Dense(1)(inputs)
# Define the Dense layer and connect it to the input
outputs = keras.layers.Dense(1)(x)


# Create the functional model
model = keras.Model(inputs=inputs, outputs=outputs)
model(X) # build


tx = optax.sgd(1e-3)
trainable_var = nnx.All(keras.Variable, lambda path, x: getattr(x, '_trainable', False))
optimizer = nnx.Optimizer(model, tx, wrt=trainable_var)


@nnx.jit
def train_step(model, optimizer, batch):
 x, y = batch


 def loss_fn(model_):
   y_pred = model_(x)
   return jnp.mean((y - y_pred) ** 2)


 diff_state = nnx.DiffState(0, trainable_var)
 grads = nnx.grad(loss_fn, argnums=diff_state)(model)
 jax.debug.print("kernel(1) grad-->{}", grads._layers[1]._kernel.value)
 jax.debug.print("bias (1) grad-->{}", grads._layers[1]._trainable_variables[1].value)
 optimizer.update(grads)


@nnx.jit
def test_step(model, batch):
 x, y = batch
 y_pred = model(x)
 loss = jnp.mean((y - y_pred) ** 2)
 return {'loss': loss}




def dataset(batch_size=10):
 while True:
   idx = np.random.choice(len(X), size=batch_size)
   yield X[idx], Y[idx]


for step, batch in enumerate(dataset()):
 train_step(model, optimizer, batch)


 if step % 10 == 0:
   logs = test_step(model, (X, Y))
   print(f"step: {step}, loss: {logs['loss']}")


 if step >= 10:
   break
