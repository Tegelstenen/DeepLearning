import numpy as np

from src.utils import init_network, soft_max, cross_entropy

class Network:
    
    def __init__(self, network_shape) -> None:
        self.transitions = init_network(network_shape)
    
    def forward(self, X, activation=lambda X: np.maximum(0, X)):
        for trans in self.transitions:
            W = trans['W']
            b = trans['b']
            S = np.matmul(W, X) + b
            if trans == self.transitions[-1]:  # Check if this is the last layer
                P = soft_max(S)
                return P
            else:
                X = activation(S)
                
    def backward(self, X, Y, P, lam):
        grads = {}
        n_b = X.shape[1]
        G = -(Y - P)
        grads['W'] = 1/n_b * np.matmul(G, X.T) + 2*lam*self.transitions[0]["W"] # ! only handles 1 dim
        grads['b'] = 1/n_b * np.sum(G, axis=1, keepdims=True)
        return grads
                
    def loss(self, P, Y):
        return np.mean(cross_entropy(P, Y))
    
    def accuracy(self, X, y):
        P = self.forward(X)
        y_pred = np.argmax(P, axis=0)
        return np.count_nonzero(y_pred == y) / y_pred.shape[0]
    
    def train(self, X_train, Y_train, X_val, Y_val, n=100, lr=.001, epochs=40, lam=0, seed=42):
        from tqdm import tqdm
        
        rng = np.random.default_rng()    
        BitGen = type(rng.bit_generator)
        rng.bit_generator.state = BitGen(seed).state
        N = X_train.shape[1]
        
        losses = {
            "train" : {},
            "val" : {}
        }
        
        costs = {
            "train" : {},
            "val" : {}
        }
        
        # Initial loss
        P_train = self.forward(X_train)
        losses["train"][0] = self.loss(P_train , Y_train)
        costs["train"][0] = losses["train"][0] + lam*np.sum(self.transitions[0]['W']**2) #! only for 1 dim
        
        P_val = self.forward(X_val)
        losses["val"][0] = self.loss(P_val , Y_val)
        costs["val"][0] = losses["val"][0] + lam*np.sum(self.transitions[0]['W']**2) #! only for 1 dim
        
        for ep in tqdm(range(epochs), desc="Training epochs"):
            indices = rng.permutation(N)
            X_perm = X_train[:, indices]
            Y_perm = Y_train[:, indices]
            
            for j in range(int(N/n)):
                start = j*n
                end = (j+1)*n
                X_batch = X_perm[:, start:end]
                Y_batch = Y_perm[:, start:end]
                
                P_batch = self.forward(X_batch)
                grads = self.backward(X_batch, Y_batch, P_batch, lam)        
                
                self.transitions[0]['W'] = self.transitions[0]['W'] - lr * grads['W'] #! Will only wokr for 1 layer atm
                self.transitions[0]['b'] = self.transitions[0]['b'] - lr * grads['b'] #! Will only wokr for 1 layer atm
            
            P_train = self.forward(X_train)
            losses["train"][ep+1] = self.loss(P_train , Y_train)
            costs["train"][ep+1] = losses["train"][ep+1] + lam*np.linalg.norm(self.transitions[0]['W'], 2) #! only for 1 dim
            
            P_val = self.forward(X_val)
            losses["val"][ep+1] = self.loss(P_val , Y_val)
            costs["val"][ep+1] = losses["val"][ep+1] + lam*np.linalg.norm(self.transitions[0]['W'], 2) #! only for 1 dim
            
            tqdm.write(f"Epoch {ep+1}/{epochs} - Train loss: {losses['train'][ep+1]:.4f}, Val loss: {losses['val'][ep+1]:.4f}")
        
        return losses, costs