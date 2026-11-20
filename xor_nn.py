"""
The SAME network and formulas as addition_nn_numpy.py, now on XOR.

XOR is not linearly separable: no straight line splits {(0,1), (1,0)} from
{(0,0), (1,1)}. With identity activation the whole net collapses to one linear map
(y_hat = W2 @ W1 @ x) and cannot learn it. A sigmoid hidden layer can.

Same 2-3-1 architecture, still no bias -- only the activation f changes.
"""

import matplotlib.pyplot as plt
import numpy as np

import addition_nn_numpy as nn

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

EPOCHS = 5000
ALPHA = 0.1
SEED = 0


def train(activation):
    nn.ACTIVATION = activation  # f / f_prime read this module global at call time
    rng = np.random.default_rng(SEED)
    W1 = rng.standard_normal((nn.H, 2))
    W2 = rng.standard_normal((1, nn.H))
    history = []
    for _ in range(EPOCHS):
        cost = 0.0
        for i in range(len(X)):
            x, y = X[i].reshape(2, 1), Y[i].reshape(1, 1)
            z1, a1, z2, a2 = nn.forward(x, W1, W2)
            dC_dW1, dC_dW2 = nn.backward(x, y, z1, a1, z2, a2, W2)
            W2 = W2 + ALPHA * dC_dW2
            W1 = W1 + ALPHA * dC_dW1
            cost += 0.5 * float(((y - a2) ** 2).item())
        history.append(cost / len(X))
    preds = [float(nn.forward(X[i].reshape(2, 1), W1, W2)[3].item()) for i in range(len(X))]
    return history, preds


def main():
    results = {act: train(act) for act in ("identity", "sigmoid")}

    for act, (history, preds) in results.items():
        print(f"f = {act:8s} | final cost = {history[-1]:.6f}")
        for (a, b), p, y in zip(X.astype(int), preds, Y[:, 0].astype(int)):
            print(f"  {a} XOR {b} = {p:6.3f}   (correct: {y})")

    plt.figure(figsize=(7, 4))
    for act, (history, _) in results.items():
        plt.plot(history, label=f"hidden f = {act}")
    plt.xlabel("epoch")
    plt.ylabel("mean cost C")
    plt.yscale("log")
    plt.title("XOR: linear net stalls, sigmoid hidden layer learns it")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/xor_learning_curve.png", dpi=150)
    print("saved assets/xor_learning_curve.png")


if __name__ == "__main__":
    main()
