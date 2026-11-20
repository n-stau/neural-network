"""
A neural network learns addition (2 + 2) -- with KERAS (TensorFlow).

Same architecture as addition_nn_numpy.py:
    2 inputs -> hidden layer with 3 neurons -> 1 output neuron, no bias.

THE CONTRAST to the from-scratch version:
Keras does feedforward AND backpropagation internally. The formulas (delta,
Hadamard product, W2.T, update rule) are NOT visible here -- `model.fit(...)`
does all of it. That is why the from-scratch file exists, where every formula
is spelled out.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # less TensorFlow log noise

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------- Training data -------------------------------------
X = np.random.uniform(0, 10, size=(2000, 2))     # 2000 pairs of numbers
Y = X.sum(axis=1, keepdims=True)                 # y = x1 + x2

# ------------------------------------- Model (Keras) -----------------------------------
# Sequential model: layers are simply stacked.
# use_bias=False -> no bias, same as the from-scratch net.
model = keras.Sequential([
    keras.Input(shape=(2,)),                               # 2 inputs
    layers.Dense(3, activation="linear", use_bias=False),  # hidden layer, 3 neurons
    layers.Dense(1, activation="linear", use_bias=False),  # 1 output neuron
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss="mse")
model.summary()

# ------------------------------------- Training ----------------------------------------
print("\nTraining ...")
model.fit(X, Y, epochs=200, batch_size=32, verbose=0)

# -------------------------------------- Tests ------------------------------------------
print("\nTesting the trained Keras net:")
for a, b in [(2, 2), (3, 5), (7, 1), (0, 0), (9, 6)]:
    pred = float(model.predict(np.array([[a, b]], dtype=float), verbose=0)[0, 0])
    print(f"  {a} + {b} = {pred:6.3f}   (correct: {a + b})")

pred_22 = float(model.predict(np.array([[2, 2]], dtype=float), verbose=0)[0, 0])
print(f"\n=> Keras result for 2 + 2: {pred_22:.4f}  (expected ~ 4.0)")
