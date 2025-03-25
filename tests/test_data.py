import numpy as np
import pytest
from src.data import load_batch, get_moments, normalize

@pytest.fixture
def cifar_data():
    cifar_dir = "datasets/cifar-10-batches-py"
    filename = cifar_dir + "/data_batch_1"
    X, Y, y = load_batch(filename)
    return X, Y, y

def test_image_dimensions(cifar_data):
    X, _, _ = cifar_data
    d, n = X.shape
    assert d == 32*32*3, f"Wrong image dimensionality {d}"
    assert n == 10000, f"Wrong num of images: {n}"

def test_image_value_range(cifar_data):
    X, _, _ = cifar_data
    assert not np.any(X < 0.0), "X contains values < 0.0"
    assert not np.any(X > 1.0), "X contains values > 1.0"

def test_label_dimensions(cifar_data):
    _, _, y = cifar_data
    assert y.size == 10000, "Wrong num of labels"

def test_label_values(cifar_data):
    _, _, y = cifar_data
    valid_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert not np.any(np.isin(y, valid_labels, invert=True)), "Y contains invalid class labels"

def test_one_hot_encoding(cifar_data):
    _, Y, _ = cifar_data
    K, n = Y.shape
    assert K == 10, f"Wrong amount of one-hot encodings: {K}"
    assert n == 10000, "Wrong num of labels"
    # Additional one-hot encoding tests
    assert np.all(np.sum(Y, axis=0) == 1), "Each sample should have exactly one class"
    assert np.all(np.isin(Y, [0, 1])), "One-hot encoding should only contain 0s and 1s"
    
def test_pre_process(cifar_data):
    X, _, _ = cifar_data
    mean_X, std_X = get_moments(X)
    X_norm = normalize(X, mean_X, std_X)
    
    # Test that mean_X has the right shape
    d = X.shape[0]
    assert mean_X.shape == (d, 1), f"Mean should have shape ({d}, 1), got {mean_X.shape}"
    
    # Test that std_X has the right shape
    assert std_X.shape == (d, 1), f"Std should have shape ({d}, 1), got {std_X.shape}"
    
    # Test that normalized data has same shape as input
    assert X_norm.shape == X.shape, f"Normalized data shape {X_norm.shape} doesn't match input shape {X.shape}"
    
    # Test that normalized data has approximately zero mean
    assert np.allclose(np.mean(X_norm, axis=1), 0, atol=1e-5), "Normalized data should have zero mean"
    
    # Test that normalized data has approximately unit variance
    assert np.allclose(np.std(X_norm, axis=1), 1, atol=1e-5), "Normalized data should have unit variance"
    
    # Test that normalization is correctly using broadcasting
    X_manual = (X - mean_X) / std_X
    assert np.allclose(X_norm, X_manual), "Normalization should be equivalent to (X - mean_X) / std_X with broadcasting"