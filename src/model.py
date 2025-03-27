import numpy as np
from tqdm.notebook import tqdm


from src.utils import init_network, soft_max, cross_entropy, flip_augment, multiple_cross_entropy

class Network:
    
    def __init__(self, network_shape, final_layer=soft_max, loss_function=cross_entropy) -> None:
        self.transitions = init_network(network_shape)
        self.final_layer = final_layer
        self.loss_function = loss_function
    
    def forward(self, X, activation=lambda X: np.maximum(0, X)):
        for trans in self.transitions:
            W = trans['W']
            b = trans['b']
            S = np.matmul(W, X) + b
            if trans == self.transitions[-1]:
                P = self.final_layer(S)
                return P
            else:
                X = activation(S)
                
    def backward(self, X, Y, P, lam):
        grads = {}
        n_b = X.shape[1]
        if self.loss_function is cross_entropy:
            G = -(Y - P)
        elif self.loss_function is multiple_cross_entropy:
            K = P.shape[0]
            G = -(Y - P)/K
        grads['W'] = 1/n_b * np.matmul(G, X.T) + 2*lam*self.transitions[0]["W"] # ! only handles 1 dim
        grads['b'] = 1/n_b * np.sum(G, axis=1, keepdims=True)
        return grads
                
    def loss(self, P, Y):
        return np.mean(self.loss_function(P, Y))
    
    def accuracy(self, X, y):
        P = self.forward(X)
        y_pred = np.argmax(P, axis=0)
        return np.count_nonzero(y_pred == y) / y_pred.shape[0]
    
    def train(self, X_train, Y_train, X_val, Y_val, n_batch=100, lr=.001, epochs=40, lam=0, seed=42, augmentation=False, decay_rate=0.1, decay_steps=10, patience=5):
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
        
        P_train = self.forward(X_train)
        losses["train"][0] = self.loss(P_train, Y_train)
        costs["train"][0] = losses["train"][0] + lam*np.sum(self.transitions[0]['W']**2)
        
        P_val = self.forward(X_val)
        losses["val"][0] = self.loss(P_val, Y_val)
        costs["val"][0] = losses["val"][0] + lam*np.sum(self.transitions[0]['W']**2)
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        initial_lr = lr
        
        # Create progress bar with initial description
        pbar = tqdm(range(epochs), desc=f"Training | lam={lam:.0e}, lr={lr:.4f}, batch_size={n_batch}, decay_steps={decay_steps}")
        
        for ep in pbar:
            # Apply step decay
            if decay_steps > 0 and ep > 0 and ep % decay_steps == 0:
                lr = initial_lr * (decay_rate ** (ep // decay_steps))
            
            if augmentation:
                X_train_copy = flip_augment(X_train.copy())
            else:
                X_train_copy = X_train.copy()
                
            indices = rng.permutation(N)
            X_perm = X_train_copy[:, indices]
            Y_perm = Y_train[:, indices]
            
            for j in range(int(N/n_batch)):
                start = j*n_batch
                end = (j+1)*n_batch
                X_batch = X_perm[:, start:end]
                Y_batch = Y_perm[:, start:end]
                
                P_batch = self.forward(X_batch)
                grads = self.backward(X_batch, Y_batch, P_batch, lam)        
                
                self.transitions[0]['W'] = self.transitions[0]['W'] - lr * grads['W']
                self.transitions[0]['b'] = self.transitions[0]['b'] - lr * grads['b']
            
            P_train = self.forward(X_train)
            losses["train"][ep+1] = self.loss(P_train, Y_train)
            
            P_val = self.forward(X_val)
            losses["val"][ep+1] = self.loss(P_val, Y_val)
            
            # Update progress bar with all information in a single line
            pbar.set_postfix({
                'Epoch': f"{ep+1}/{epochs}",
                'Train': f"{losses['train'][ep+1]:.4f}",
                'Val': f"{losses['val'][ep+1]:.4f}",
                'lr': f"{lr:.6f}"
            })
            
            # Early stopping check
            current_val_loss = losses["val"][ep+1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights = [{
                    'W': trans['W'].copy(),
                    'b': trans['b'].copy()
                } for trans in self.transitions]
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                pbar.set_description(f"Early stopped at epoch {ep+1}")
                self.transitions = best_weights
                break
        
        pbar.close()
        return losses
    
    
    
    
    
    