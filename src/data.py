import numpy as np
import matplotlib.pyplot as plt

import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_meta_data(filename):
    file_dict = unpickle(filename)
    return file_dict[b'label_names']
    

def load_batch(filename):
    # Load a batch of training data
    file_dict = unpickle(filename)

    # ---------- Extract the image data
    X = file_dict[b"data"].astype(np.float64) / 255.0
    X = X.transpose()
    
    # ---------- Extract the image labels
    y = np.array(file_dict[b"labels"]).astype(np.float64)
    
    # ---------- Extract the label encodings
    # One-hot encode the labels
    Y = np.zeros((10, y.size))
    for i in range(y.size):
        Y[int(y[i]), i] = 1
    
    return X, Y, y

def make_printable(X):
    # Reshape each image from a column vector to a 3d array
    nn = X.shape[1]
    X_im = X.reshape((32, 32, 3, nn), order="F")
    X_im = np.transpose(X_im, (1, 0, 2, 3))
    
    return X_im

def get_moments(X):
    d = X.shape[0]
    mean = np.mean(X, axis=1).reshape(d, 1)
    std = np.std(X, axis=1).reshape(d, 1)
    return mean, std

def normalize(X, mean, std):
    return (X - mean)/std

def load_all_train_data():
    dir = "datasets/cifar-10-batches-py/"
    X_all = None
    Y_all = None
    y_all = None
    for i in range(1,6):
        file = dir+f"data_batch_{i}"
        X_batch, Y_batch, y_batch = load_batch(file)
        if X_all is None:
            X_all = X_batch
            Y_all = Y_batch
            y_all = y_batch
        else:
            X_all = np.hstack((X_all, X_batch))
            Y_all = np.hstack((Y_all, Y_batch))
            y_all = np.hstack((y_all, y_batch))
    
    return X_all, Y_all, y_all
