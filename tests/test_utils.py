import numpy as np
import pytest
from src.utils import init_network, soft_max, relative_error, flip_augment

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
    
    
    def test_flip_augment():
        # Create a simple test image as a 3D array (channels, height, width)
        test_image = np.array([
            [[1, 2, 3],
            [4, 5, 6]],  # Channel 1
            [[7, 8, 9],
            [10, 11, 12]],  # Channel 2
            [[13, 14, 15],
            [16, 17, 18]]   # Channel 3
        ])
        
        # Expected result after horizontal flip
        expected_flip = np.array([
            [[3, 2, 1],
            [6, 5, 4]],  # Channel 1
            [[9, 8, 7],
            [12, 11, 10]],  # Channel 2
            [[15, 14, 13],
            [18, 17, 16]]   # Channel 3
        ])
        
        # Create a batch of images (2 images)
        X = np.stack([test_image, test_image], axis=1)  # Shape: (channels, n_samples, height, width)
        
        # Test flip augmentation
        X_aug = flip_augment(X)
        
        # Test shape hasn't changed
        assert X_aug.shape == X.shape
        
        # Test that some images are flipped (since it's random, we'll run multiple times)
        n_trials = 100
        n_flips = 0
        
        for _ in range(n_trials):
            X_aug = flip_augment(X)
            # Check if any images in batch were flipped
            for i in range(X.shape[1]):
                if np.array_equal(X_aug[:, i], expected_flip):
                    n_flips += 1
        
        # With p=0.5, expect roughly half of images to be flipped
        flip_rate = n_flips / (n_trials * X.shape[1])
        assert 0.4 < flip_rate < 0.6, f"Flip rate {flip_rate} is outside expected range"
        
        # Test that original data is unchanged
        X_copy = X.copy()
        _ = flip_augment(X)
        assert np.array_equal(X, X_copy), "Original data was modified"
        