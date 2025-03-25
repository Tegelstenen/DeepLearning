import numpy as np
import pytest
from src.utils import init_network, soft_max, relative_error

def test_init_network_shapes():
    # Test different network architectures
    network_shapes = [
        [2, 3],  # Simple 2->3 network
        [4, 5, 2],  # Three layer network
        [10, 8, 6, 4]  # Deep network
    ]
    
    for shape in network_shapes:
        network = init_network(shape)
        
        # Check if we have the correct number of layers
        assert len(network) == len(shape) - 1
        
        # Check each layer's shapes
        for i in range(len(shape) - 1):
            layer = network[i]
            
            # Check if W and b exist
            assert 'W' in layer
            assert 'b' in layer
            
            # Check shapes
            assert layer['W'].shape == (shape[i+1], shape[i])
            assert layer['b'].shape == (shape[i+1], 1)

def test_init_network_values():
    # Test with a simple network
    shape = [3, 4]
    network = init_network(shape)
    layer = network[0]
    
    # Check if biases are initialized to zero
    assert np.allclose(layer['b'], 0)
    
    # Check if weights follow normal distribution with std=0.01
    W = layer['W']
    assert abs(np.std(W) - 0.01) < 0.005  # Allow some deviation due to random initialization
    assert abs(np.mean(W)) < 0.005  # Mean should be close to 0

def test_soft_max():
    # Test single sample
    x = np.array([[1.0], [2.0], [3.0]])
    output = soft_max(x)
    
    # Check shape
    assert output.shape == x.shape
    
    # Check sum to 1
    assert np.isclose(np.sum(output), 1.0)
    
    # Check ordering preserved
    assert np.argmax(x) == np.argmax(output)
    
    # Test batch processing
    batch = np.array([[1.0, 4.0], 
                     [2.0, 5.0], 
                     [3.0, 6.0]])
    batch_output = soft_max(batch)
    
    # Check batch shape
    assert batch_output.shape == batch.shape
    
    # Check each sample sums to 1
    assert np.allclose(np.sum(batch_output, axis=0), 1.0)
    
    # Test numerical stability with large numbers
    large_nums = np.array([[100.0], [100.1], [100.2]])
    large_output = soft_max(large_nums)
    
    # Should still sum to 1 and preserve ordering
    assert np.isclose(np.sum(large_output), 1.0)
    assert np.argmax(large_nums) == np.argmax(large_output)
    
    
def test_relative_error():
    Ga = {
        'W': np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]]),
        'b': np.array([[0.1], [0.2], [0.3]])
    }
    
    Gn = {
        'W': np.array([[0.1000001, 0.2000001, 0.300001],
              [0.4000001, 0.5000001, 0.6000001]]),
        'b': np.array([[0.1000001], [0.2000001], [0.3000001]])
    }

    
    # Test overall
    assert relative_error(Gn, Ga), "Relative error check failed for the complete gradient"
    