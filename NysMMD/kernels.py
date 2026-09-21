import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

'''def RBFkernel(X1: np.ndarray, X2: np.ndarray, sigma: float) -> tf.tensor:   ##(n_samples_1, n_samples_2)
    """
    Computes the Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.
    The function calculates the kernel matrix using the formula:
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        X1 (np.ndarray): Input data of shape (n_samples_1, n_features).
        X2 (np.ndarray): Input data of shape (n_samples_2, n_features).
        sigma (float): The standard deviation parameter for the Gaussian kernel.

    Returns:
        np.ndarray: The computed kernel matrix of shape (n_samples_1, n_samples_2).
    """
    gamma = 1 / (2 * sigma**2)  # Convert sigma to gamma for rbf_kernel computation
    return rbf_kernel(X1, X2, gamma)'''
    
def RBFkernel_tf(X1, X2, sigma):
    gamma = 1.0 / (2.0 * sigma**2)

    # ||x||^2
    X1_sq = tf.reduce_sum(tf.square(X1), axis=1, keepdims=True)   # (n, 1)
    X2_sq = tf.reduce_sum(tf.square(X2), axis=1, keepdims=True)   # (m, 1)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    dist_sq = X1_sq - 2.0 * tf.matmul(X1, X2, transpose_b=True) + tf.transpose(X2_sq)

    return tf.exp(-gamma * dist_sq)