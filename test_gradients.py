"""
Gradient check: the hand-written backprop in addition_nn_numpy.py must match a
numerical derivative of the cost C = 1/2 (y - y_hat)^2.

backward() returns -dC/dW (the minus sign is folded in, see its docstring), so we
compare it against the negative central difference. Run: python test_gradients.py
(also works with pytest).
"""

import numpy as np

import addition_nn_numpy as nn

EPS = 1e-6


def cost(x, y, W1, W2):
    a2 = nn.forward(x, W1, W2)[3]
    return 0.5 * float(((y - a2) ** 2).item())


def numerical_grad(W, x, y, W1, W2):
    """Central difference dC/dW for every entry of W (W is W1 or W2, edited in place)."""
    grad = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        old = W[idx]
        W[idx] = old + EPS
        c_plus = cost(x, y, W1, W2)
        W[idx] = old - EPS
        c_minus = cost(x, y, W1, W2)
        W[idx] = old
        grad[idx] = (c_plus - c_minus) / (2 * EPS)
    return grad


def check(activation):
    nn.ACTIVATION = activation
    rng = np.random.default_rng(1)
    W1 = rng.standard_normal((nn.H, 2))
    W2 = rng.standard_normal((1, nn.H))
    x = rng.uniform(-2, 2, size=(2, 1))
    y = rng.uniform(-2, 2, size=(1, 1))

    dC_dW1, dC_dW2 = nn.backward(x, y, *nn.forward(x, W1, W2), W2)
    np.testing.assert_allclose(dC_dW1, -numerical_grad(W1, x, y, W1, W2), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dC_dW2, -numerical_grad(W2, x, y, W1, W2), rtol=1e-5, atol=1e-8)


def test_identity():
    check("identity")


def test_sigmoid():
    check("sigmoid")


if __name__ == "__main__":
    test_identity()
    test_sigmoid()
    print("gradient check passed (identity + sigmoid)")
