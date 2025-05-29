import keras as k
import keras.layers as kl
from keras import models as km

def runTest():
    il = kl.Input((1024, 4))
    cl = kl.Conv1D(filters=5, kernel_size=10, padding="valid", activation="relu")(il)
    m = km.Model(inputs=il, outputs=[cl])
    print(m.input.shape)
    m.save("test.keras")

def runLoad():
    m = km.load_model("test.keras")
    print(type(m.inputs[0]))
    print(m.inputs[0].shape)

runTest()
runLoad()
