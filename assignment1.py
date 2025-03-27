import matplotlib.pyplot as plt
import numpy as np
import os

from src.data import load_batch, get_moments, normalize, load_meta_data
from src.model import Network
from src.utils import relative_error
from src.torch_gradient_computations import ComputeGradsWithTorch


# ---------- Get data ----------
X_train, Y_train, y_train = load_batch("datasets/cifar-10-batches-py/data_batch_1")
X_val, Y_val, y_val = load_batch("datasets/cifar-10-batches-py/data_batch_2")
X_test, Y_test, y_test = load_batch("datasets/cifar-10-batches-py/test_batch")

# ---------- Preprocess ----------
mean_X, std_X = get_moments(X_train)
X_train = normalize(X_train, mean_X, std_X)
X_val = normalize(X_val, mean_X, std_X)
X_test = normalize(X_test, mean_X, std_X)

# ---------- intantiate model ----------
input_dim = X_train.shape[0]
output_dim = Y_train.shape[0]
model = Network([input_dim, output_dim])


# ---------- test functions ----------
P_train = model.forward(X_train[:, 0:100])

loss = model.loss(P_train, Y_train[:, 0:100])
print(f"loss {loss}")

acc = model.accuracy(X_train[:, 0:100], y_train[0:100])
print(f"accuracy {acc}")

n_small = 1
lam = 0.1
X_small = X_train[:,0:n_small]
Y_small = Y_train[:,0:n_small]
y_small = y_train[0:n_small]
P_small = model.forward(X_small)
my_grads = model.backward(X_small, Y_small, P_small, lam)
torch_grads = ComputeGradsWithTorch(X_small, y_small, model.transitions[0], lam)

print("the grads are the same: ", relative_error(my_grads, torch_grads))


# ---------- train test network ----------
accs = []
params = [
    (0, .1),
    (0, .001),
    (.1, .001),
    (1, .001),
]


os.makedirs("img/ass1/w", exist_ok=True)
os.makedirs("img/ass1/loss", exist_ok=True)
label_names = load_meta_data("datasets/cifar-10-batches-py/batches.meta")

for lam, lr in params:
    model = Network([input_dim, output_dim])
    
    losses, costs = model.train(X_train, Y_train, X_val, Y_val, lr=lr, lam=lam)
    accs.append(model.accuracy(X_test, y_test))
    plt.plot(losses["train"].keys(), losses["train"].values(), label="train")
    plt.plot(losses["val"].keys(), losses["val"].values(), label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"loss at lam={lam} and lr={lr}")
    plt.legend()
    plt.savefig(f"img/ass1/loss/l_lam_{lam}_lr_{lr}.png")
    plt.close()

    if lam > 0:
        plt.plot(costs["train"].keys(), costs["train"].values(), label="train")
        plt.plot(costs["val"].keys(), costs["val"].values(), label="val")
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.title(f"cost at lam={lam} and lr={lr}")
        plt.legend()
        plt.savefig(f"img/ass1/loss/c_lam_{lam}_lr_{lr}.png")
        plt.close()

    # ---------- visualize weights --------
    Ws = model.transitions[0]['W'].transpose().reshape((32,32,3,10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    fig.suptitle(f"lam={lam} and lr={lr}")
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        axes[i].imshow(w_im_norm)
        axes[i].set_title(label_names[i].decode('utf-8'))
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"img/ass1/w/lam_{lam}_lr_{lr}.png")
    plt.close()
    
print([acc * 100 for acc in accs])
