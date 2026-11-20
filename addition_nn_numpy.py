"""
A neural network learns addition (2 + 2) -- FROM SCRATCH in NumPy.

This script implements feedforward and backpropagation exactly as they are derived
by hand in a lecture. Variable names match the math symbols, so every formula can
be recognised in the code:

    z1, a1, z2, a2, W1, W2, delta1, delta2, dC_dW1, dC_dW2

Architecture:
    Input  x  in R^(2x1)              (2 inputs, NO bias)
    W1        in R^(3x2)   ->  hidden layer with 3 neurons,  a1 in R^(3x1)
    W2        in R^(1x3)   ->  1 output neuron,              a2 = y_hat in R^(1x1)
    Cost:     C = 1/2 (y - y_hat)^2
"""

import numpy as np

# ====================== KNOBS (change these and watch the effect) =====================
#
# >>> How much training data?  Try it -- this answers "memorise or learn the rule": <<<
N_SAMPLES     = 4            # number of training examples.
                            #   1   -> the net MEMORISES (test cost stays high!)
                            #   2-3 -> the rule is learned (test cost ~ 0)
                            #   200 -> the usual case, learns the rule
N_TEST        = 200         # held-out test examples (NEVER trained on)
#
# >>> How long to train?  Epochs = how many times all training data is run through. <<<
EPOCHS        = 10000        # more epochs = more learning steps. See README: epochs are
                            # needed because learning is iterative -- in ONE pass the
                            # net does not learn the task.
#
# >>> Architecture & learning <<<
HIDDEN_NEURONS = 3          # neurons in the hidden layer
ACTIVATION     = "identity" # hidden activation f: "identity" (robust, clean 2+2)
                            # or "sigmoid" (needs more epochs; see xor_nn.py)
ALPHA          = 0.001      # learning rate / step size (start value). Too big -> diverges (NaN)
ALPHA_DECAY    = True       # shrink alpha over the epochs
NUMBER_RANGE   = (0, 10)    # range of the training numbers (2+2 is inside)
SEED           = 0          # for reproducible random numbers
# =======================================================================================


# ------------------------------- Activation function f ---------------------------------
# Generic f(z) with derivative f'(z), used for the HIDDEN layer:
def f(z):
    if ACTIVATION == "identity":
        return z                      # f(z) = z
    elif ACTIVATION == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown ACTIVATION: {ACTIVATION}")


def f_prime(z):
    if ACTIVATION == "identity":
        return np.ones_like(z)        # f'(z) = 1
    elif ACTIVATION == "sigmoid":
        s = f(z)
        return s * (1.0 - s)          # f'(z) = f(z) (1 - f(z))
    raise ValueError(f"Unknown ACTIVATION: {ACTIVATION}")


# H = number of hidden neurons. W1: Hx2, W2: 1xH (H=3 in the default setup).
H = HIDDEN_NEURONS


# ------------------------------------ Feedforward --------------------------------------
def forward(x, W1, W2):
    """x: (2x1) -> returns all intermediate values."""
    z1 = W1 @ x          # W1: Hx2 , x: 2x1   ->  z1: Hx1
    a1 = f(z1)           #                        a1: Hx1   (hidden layer with f)
    z2 = W2 @ a1         # W2: 1xH , a1: Hx1  ->  z2: 1x1
    a2 = z2              # OUTPUT is linear (f(z)=z), so sums > 1 can be represented
    return z1, a1, z2, a2


# ---------------------------------- Backpropagation ------------------------------------
def backward(x, y, z1, a1, z2, a2, W2):
    """Chain rule + Hadamard product (*), exactly as derived by hand."""
    # Derivative of the cost w.r.t. the output.
    # Note: we define dC_da2 = (y - a2). That makes the "+" in the update rule
    # (W -> W + alpha * dC_dW) correct: the minus sign from d/dy_hat [1/2 (y-y_hat)^2]
    # is folded in here. Mathematically identical to  W -= alpha*grad  with (a2 - y).
    dC_da2 = (y - a2)                          # 1x1

    delta2 = 1.0 * dC_da2                      # delta2 = f'(z2) (*) dC_da2, linear output -> f'=1
    dC_dW2 = delta2 @ a1.T                     # 1x1 . 1xH           -> dC_dW2: 1xH  (= shape of W2)

    delta1 = f_prime(z1) * (W2.T @ delta2)     # (Hx1) (*) (Hx1.1x1) -> delta1: Hx1
    dC_dW1 = delta1 @ x.T                      # Hx1 . 1x2           -> dC_dW1: Hx2  (= shape of W1)

    return dC_dW1, dC_dW2


# ----------------------------------- Training data -------------------------------------
def make_data(n, low, high, rng):
    """Creates n examples: X (pairs of numbers) and y = x1 + x2."""
    X = rng.uniform(low, high, size=(n, 2))    # n rows, 2 numbers each
    Y = X.sum(axis=1, keepdims=True)           # y = x1 + x2
    return X, Y


# ------------------------------------- Training ----------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    # Initialise weights (small random values). NO bias.
    W1 = rng.standard_normal((H, 2)) * 0.1     # Hx2
    W2 = rng.standard_normal((1, H)) * 0.1     # 1xH

    # Two separate data sets:
    #   X_train -> used for training
    #   X_test  -> NEVER trained on, only checked at the end (shows: real rule instead
    #              of memorising). Same range, different numbers.
    X_train, Y_train = make_data(N_SAMPLES, *NUMBER_RANGE, rng)
    X_test,  Y_test  = make_data(N_TEST,    *NUMBER_RANGE, rng)

    def mean_cost(X, Y):
        """Mean cost C over a data set (without training)."""
        total = 0.0
        for i in range(len(X)):
            _, _, _, a2 = forward(X[i].reshape(2, 1), W1, W2)
            total += 0.5 * float(((Y[i].reshape(1, 1) - a2) ** 2).item())
        return total / len(X)

    print(f"activation f = {ACTIVATION} | start learning rate alpha = {ALPHA} | epochs = {EPOCHS}")
    print(f"training data: {N_SAMPLES} examples | test data: {N_TEST} (held out)")
    print("-" * 60)

    for epoch in range(EPOCHS):
        # Optionally shrink the learning rate over time
        alpha = ALPHA / (1.0 + epoch / 1000.0) if ALPHA_DECAY else ALPHA

        for i in range(N_SAMPLES):
            x = X_train[i].reshape(2, 1)       # one input vector x: 2x1
            y = Y_train[i].reshape(1, 1)       # target output:      1x1

            # 1) Feedforward
            z1, a1, z2, a2 = forward(x, W1, W2)

            # 2) Backpropagation -> gradients
            dC_dW1, dC_dW2 = backward(x, y, z1, a1, z2, a2, W2)

            # 3) Update rule: W -> W + alpha * dC_dW
            W2 = W2 + alpha * dC_dW2
            W1 = W1 + alpha * dC_dW1

        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            # Show training AND test cost: if they converge together -> no memorising
            print(f"epoch {epoch:5d} | train cost = {mean_cost(X_train, Y_train):.6f}"
                  f" | test cost = {mean_cost(X_test, Y_test):.6f}")

    # ------------------------------------ Tests ---------------------------------------
    print("-" * 60)

    def predict(a, b):
        x = np.array([[a], [b]], dtype=float)  # 2x1
        _, _, _, a2 = forward(x, W1, W2)
        return float(a2.item())

    # Held-out test data: the net has NEVER seen these numbers during training.
    train_cost = mean_cost(X_train, Y_train)
    test_cost  = mean_cost(X_test, Y_test)
    print(f"train cost (seen):       {train_cost:.8f}")
    print(f"test cost  (never seen): {test_cost:.8f}")
    if test_cost > 10 * train_cost + 0.01:
        print("-> test cost much higher than train cost: the net MEMORISED!")
        print(f"   (too little data -- try N_SAMPLES > {N_SAMPLES})")
    else:
        print("-> test cost just as low: the net learned the RULE, not the examples.")
    print("-" * 60)

    print("Examples (inside the training range 0..10):")
    for a, b in [(2, 2), (3, 5), (7, 1), (0, 0), (9, 6)]:
        print(f"  {a} + {b} = {predict(a, b):7.3f}   (correct: {a + b})")

    # Extrapolation: numbers FAR outside 0..10. A memoriser would fail here;
    # a real rule keeps computing correctly (with identity activation).
    print("\nExtrapolation (outside the training range):")
    for a, b in [(50, 50), (100, 7), (-4, 9), (1000, 1000)]:
        print(f"  {a} + {b} = {predict(a, b):10.3f}   (correct: {a + b})")

    print("-" * 60)
    print("Learned weights:")
    print(f"W1 ({H}x2) =\n", np.round(W1, 3))
    print(f"W2 (1x{H}) =\n", np.round(W2, 3))
    if ACTIVATION == "identity":
        # With identity activation the net is linear: y_hat = W2 @ W1 @ x.
        # For correct addition, W2 @ W1 must be roughly [1, 1].
        print("W2 @ W1  =", np.round(W2 @ W1, 3), " (should be ~ [[1, 1]])")


if __name__ == "__main__":
    main()
