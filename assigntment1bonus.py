#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_all_train_data, load_batch
from src.model import Network
from src.utils import sigmoid, multiple_cross_entropy

np.random.seed(42)


# In[2]:


# ---------- get data ----------
X, Y, y = load_all_train_data()

N = X.shape[1]
val_indices = np.random.choice(N, size=1000, replace=False)
train_mask = np.ones(N, dtype=bool)
train_mask[val_indices] = False

X_train, X_val = X[:, train_mask], X[:, val_indices]
Y_train, Y_val = Y[:, train_mask], Y[:, val_indices]
y_train, y_val = y[train_mask], y[val_indices]

X_test, Y_test, y_test = load_batch("datasets/cifar-10-batches-py/test_batch")


# In[3]:


# ---------- grid search ----------
lams = [0.001, 0.01, 0.1]
lrs = [0.001, 0.01, 0.1]
ns_batch = [4, 8, 16]
decays_steps = [3, 5, 10]
accs = []
all_losses = {}
best_val_loss = float('inf')
best_config = None

# Store results in a structured way for the heatmap
accuracy_results = np.zeros((len(lams), len(lrs), len(ns_batch), len(decays_steps)))

for i, lam in enumerate(lams):
    for j, lr in enumerate(lrs):
        for k, n_batch in enumerate(ns_batch):
            for z, decays_step in enumerate(decays_steps):
                model = Network([X_train.shape[0], Y_train.shape[0]])
                losses = model.train(X_train, Y_train, X_val, Y_val, lam=lam, lr=lr, n_batch=n_batch, decay_steps=decays_step, augmentation=True)
                
                # Calculate accuracy and store results
                acc = model.accuracy(X_test, y_test)
                accs.append(acc)
                all_losses[f"{lam}_{lr}_{n_batch}_{decays_step}"] = losses
                accuracy_results[i, j, k, z] = acc
                
                # Track the best configuration
                final_val_loss = losses["val"][max(losses["val"].keys())]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_config_sigmoid = (lam, lr, n_batch, decays_step)


# In[4]:


# Create a list of all configurations and their accuracies
configs = []
for i, lam in enumerate(lams):
    for j, lr in enumerate(lrs):
        for k, n_batch in enumerate(ns_batch):
            for z, decays_step in enumerate(decays_steps):
                configs.append({
                    'lambda': lam,
                    'learning_rate': lr,
                    'batch_size': n_batch,
                    'decay_steps': decays_step,
                    'accuracy': accuracy_results[i, j, k, z]
                })

# Sort configurations by accuracy in descending order
sorted_configs = sorted(configs, key=lambda x: x['accuracy'], reverse=True)

# Print LaTeX table
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{|c|c|c|c|c|c|}")
print(r"\hline")
print(r"Rank & Accuracy & $\lambda$ & $\eta$ & Batch Size & Decay Steps \\")
print(r"\hline")
for i, config in enumerate(sorted_configs[:10]):
    print(f"{i+1} & {config['accuracy']:.4f} & {config['lambda']:.1e} & {config['learning_rate']:.1e} & {config['batch_size']} & {config['decay_steps']} \\\\")
    print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Top 10 configurations ranked by accuracy}")
print(r"\label{tab:top_configs}")
print(r"\end{table}")


# In[5]:


# ---------- grid search ----------
lams = [0.001, 0.01, 0.1]
lrs = [0.001, 0.01, 0.1]
ns_batch = [4, 8, 16]
decays_steps = [3, 5, 10]
accs = []
all_losses = {}
best_val_loss = float('inf')
best_config = None

# Store results in a structured way for the heatmap
accuracy_results = np.zeros((len(lams), len(lrs), len(ns_batch), len(decays_steps)))

for i, lam in enumerate(lams):
    for j, lr in enumerate(lrs):
        for k, n_batch in enumerate(ns_batch):
            for z, decays_step in enumerate(decays_steps):
                model = Network([X_train.shape[0], Y_train.shape[0]], sigmoid, multiple_cross_entropy)
                losses = model.train(X_train, Y_train, X_val, Y_val, lam=lam, lr=lr, n_batch=n_batch, decay_steps=decays_step, augmentation=True)
                
                # Calculate accuracy and store results
                acc = model.accuracy(X_test, y_test)
                accs.append(acc)
                all_losses[f"{lam}_{lr}_{n_batch}_{decays_step}"] = losses
                accuracy_results[i, j, k, z] = acc
                
                # Track the best configuration
                final_val_loss = losses["val"][max(losses["val"].keys())]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss



# In[6]:


# Create a list of all configurations and their accuracies
configs = []
for i, lam in enumerate(lams):
    for j, lr in enumerate(lrs):
        for k, n_batch in enumerate(ns_batch):
            for z, decays_step in enumerate(decays_steps):
                configs.append({
                    'lambda': lam,
                    'learning_rate': lr,
                    'batch_size': n_batch,
                    'decay_steps': decays_step,
                    'accuracy': accuracy_results[i, j, k, z]
                })

# Sort configurations by accuracy in descending order
sorted_configs = sorted(configs, key=lambda x: x['accuracy'], reverse=True)

# Print LaTeX table
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{|c|c|c|c|c|c|}")
print(r"\hline")
print(r"Rank & Accuracy & $\lambda$ & $\eta$ & Batch Size & Decay Steps \\")
print(r"\hline")
for i, config in enumerate(sorted_configs[:10]):
    print(f"{i+1} & {config['accuracy']:.4f} & {config['lambda']:.1e} & {config['learning_rate']:.1e} & {config['batch_size']} & {config['decay_steps']} \\\\")
    print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Top 10 configurations ranked by accuracy}")
print(r"\label{tab:top_configs}")
print(r"\end{table}")


# In[43]:


d = {'P': [], 'type': []}
df = pd.DataFrame(data=d)
# ---------- softmax best model ----------
model_soft = Network([X.shape[0], Y.shape[0]])
soft_losses = model_soft.train(X_train, Y_train, X_val, Y_val, n_batch=4, lr=1e-2, lam=1e-3, decay_steps=5, augmentation=True)


# ---------- sigmoid best model ----------
model_sigm = Network([X.shape[0], Y.shape[0]], sigmoid, multiple_cross_entropy)
sigm_losses = model_sigm.train(X_train, Y_train, X_val, Y_val, n_batch=4, lr=1e-2, lam=1e-3, decay_steps=10, augmentation=True)



# In[42]:


# For softmax model
P_softmax = model_soft.forward(X_test)  # (K,n)
guess_softmax = np.argmax(P_softmax, axis=0)
correct_mask_soft = guess_softmax == y_test

# Manually extract the probability for each true class
prob_true_class_soft = np.zeros(len(y_test))
for i in range(len(y_test)):
    prob_true_class_soft[i] = P_softmax[int(y_test[i]), i]

# For sigmoid model
P_sigm = model_sigm.forward(X_test)  # (K,n)
guess_sigm = np.argmax(P_sigm, axis=0)
correct_mask_sigm = guess_sigm == y_test

# Manually extract the probability for each true class
prob_true_class_sigm = np.zeros(len(y_test))
for i in range(len(y_test)):
    prob_true_class_sigm[i] = P_sigm[int(y_test[i]), i]

# Now plot the histograms
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top plot - Softmax
axes[0].hist(prob_true_class_soft[correct_mask_soft], bins=20, alpha=0.7, color='blue', label='Correct')
axes[0].hist(prob_true_class_soft[~correct_mask_soft], bins=20, alpha=0.7, color='red', label='Incorrect')
axes[0].set_title('Softmax: Probability Distribution for Ground Truth Class', fontsize=14)
axes[0].set_xlabel('Probability for Ground Truth Class')
axes[0].set_ylabel('Count')
axes[0].legend()

# Bottom plot - Sigmoid
axes[1].hist(prob_true_class_sigm[correct_mask_sigm], bins=20, alpha=0.7, color='blue', label='Correct')
axes[1].hist(prob_true_class_sigm[~correct_mask_sigm], bins=20, alpha=0.7, color='red', label='Incorrect')
axes[1].set_title('Sigmoid: Probability Distribution for Ground Truth Class', fontsize=14)
axes[1].set_xlabel('Probability for Ground Truth Class')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.show()


# In[44]:


# Plot training and validation losses
plt.figure(figsize=(12, 5))

# Plot for softmax model
plt.subplot(1, 2, 1)
train_loss_soft = list(soft_losses["train"].values())
val_loss_soft = list(soft_losses["val"].values())
epochs = range(len(train_loss_soft))
plt.plot(epochs, train_loss_soft, 'b-', label='Training loss')
plt.plot(epochs, val_loss_soft, 'r-', label='Validation loss')
plt.title('Training and Validation Loss (Softmax)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for sigmoid model
plt.subplot(1, 2, 2)
train_loss_sigm = list(sigm_losses["train"].values())
val_loss_sigm = list(sigm_losses["val"].values())
epochs = range(len(train_loss_sigm))
plt.plot(epochs, train_loss_sigm, 'b-', label='Training loss')
plt.plot(epochs, val_loss_sigm, 'r-', label='Validation loss')
plt.title('Training and Validation Loss (Sigmoid)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('loss_plots.png')
plt.show()


# In[ ]:




