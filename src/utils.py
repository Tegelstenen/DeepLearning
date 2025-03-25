import numpy as np

rng = np.random.default_rng()    
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

def init_network(network_shape):
    """
    network_shape will be a list of dimensions for each layer
    """
    transitions = []
    for i in range(len(network_shape) - 1):
        trans = {}
        from_dim = network_shape[i]
        to_dim = network_shape[i+1]
        trans['W'] = .01*rng.standard_normal(size = (to_dim, from_dim))
        trans['b'] = np.zeros((to_dim, 1))
        transitions.append(trans)

    return transitions

def soft_max(s):
    # Subtract the maximum value for numerical stability to prevent overflow
    s_shifted = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shifted)
    sum_exp = np.sum(exp_s, axis=0)
    return exp_s / sum_exp

def cross_entropy(p, y):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(y * np.log(p_safe)) / p.shape[1]  # Average over batch

def relative_error(Gn, Ga, eps=1e-4):
    for key in Ga:
        g_n = Gn[key]
        g_a = Ga[key]
        rel_err = np.abs(g_a - g_n) / np.maximum(np.abs(g_a), np.abs(g_n))
        max_err = np.max(rel_err)
        if np.any(rel_err > eps):
            print(f"Failed on key {key}")
            print(f"Maximum relative error: {max_err}")
            print(f"Epsilon threshold: {eps}")
            return False
    return True    
