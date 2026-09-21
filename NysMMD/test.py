import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

'''def RBFkernel_tf(X1, X2, sigma):
    gamma = 1.0 / (2.0 * sigma**2)

    X1_sq = tf.reduce_sum(tf.square(X1), axis=-1, keepdims=True)   # [C, n, 1]
    X2_sq = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)   # [C, m, 1]

    cross = tf.matmul(X1, X2, transpose_b=True)                    # [C, n, m]

    dist_sq = X1_sq - 2.0 * cross + tf.transpose(X2_sq, perm=[0, 2, 1])  # [C, n, m]
    dist_sq = tf.maximum(dist_sq, 0.0)

    return tf.exp(-gamma * dist_sq)'''


def median_heuristic_sigma_tf(Z, max_points=1000, eps=1e-12):
    # Z: [C, n, d]
    n = tf.shape(Z)[1]
    m = tf.minimum(n, max_points)

    inds = tf.random.shuffle(tf.range(n))[:m]
    Zs = tf.gather(Z, inds, axis=1)  # [C, m, d]

    Z_sq = tf.reduce_sum(tf.square(Zs), axis=-1, keepdims=True)  # [C, m, 1]
    dist_sq = Z_sq - 2.0 * tf.matmul(Zs, Zs, transpose_b=True) + tf.transpose(Z_sq, [0, 2, 1])
    dist_sq = tf.maximum(dist_sq, 0.0)

    # tolgo diagonale, perché è zero e falserebbe la mediana
    mask = ~tf.eye(m, dtype=tf.bool)
    dist_sq_offdiag = tf.boolean_mask(dist_sq, mask, axis=1)  # [C, m*(m-1)]

    med_dist_sq = tfp.stats.percentile(dist_sq_offdiag, 50.0, axis=-1)  # [C]
    sigma = tf.sqrt(tf.maximum(med_dist_sq, eps))                       # [C]

    return sigma



def RBFkernel_tf(X1, X2, sigma, multi_sigma=False):
    # X1: [C, n, d]
    # X2: [C, m, d]
    # sigma: scalar oppure [C]

    X1_sq = tf.reduce_sum(tf.square(X1), axis=-1, keepdims=True)
    X2_sq = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)

    cross = tf.matmul(X1, X2, transpose_b=True)
    dist_sq = X1_sq - 2.0 * cross + tf.transpose(X2_sq, [0, 2, 1])
    dist_sq = tf.maximum(dist_sq, 0.0)

    sigma = tf.cast(sigma, X1.dtype)

    if sigma.shape.rank == 1:
        sigma = sigma[:, None, None]

    
    if multi_sigma:
        sigmas = tf.stack([sigma/8, sigma/4, sigma/2, sigma, sigma*2, sigma*4],axis=0)  # [6, C, 1, 1]
        gamma = 1.0 / (2.0 * tf.square(sigmas))  # [6, C, 1, 1]
        kernels = tf.exp(-gamma * dist_sq[None, :, :, :])  # [6, C, n, k]
        return tf.reduce_mean(kernels, axis=0)  # [C, n, k]
    else:
        gamma = 1.0 / (2.0 * sigma**2)
        return tf.exp(-gamma * dist_sq)

def nystrom_features_tf(X, nys_inds, sigma, multi_sigma = False):
    # X: [C, 2b_s, d]
    # nys_inds: [k]

    centers = tf.gather(X, nys_inds, axis=1)        # [C, k, d]

    Knm = RBFkernel_tf(X, centers, sigma, multi_sigma)           # [C, n, k]
    Km  = RBFkernel_tf(centers, centers, sigma, multi_sigma)     # [C, k, k]

    eps = tf.cast(1e-6, Km.dtype)
    Km = Km + eps * tf.eye(tf.shape(Km)[-1], batch_shape=tf.shape(Km)[:-2], dtype=Km.dtype)

    S, U, V = tf.linalg.svd(Km, full_matrices=False)
    S = tf.maximum(S, 1e-12)

    Km_inv_sqrt = tf.matmul(
        U,
        tf.matmul(tf.linalg.diag(1.0 / tf.sqrt(S)), V, transpose_b=True)
    )                                               # [C, k, k]

    return tf.matmul(Knm, Km_inv_sqrt)              # [C, n, k]




def nys_inds_tf(X, k, method='uniform', sigma=1.0, seed=None):
    ntot = tf.shape(X)[0]

    k = tf.minimum(k, ntot)

    if method == 'uniform':
        inds = tf.random.shuffle(tf.range(ntot), seed=seed)[:k]
        return inds, k

    elif method == 'rlss':
        raise NotImplementedError("rlss non implementato in TensorFlow")

    else:
        raise ValueError("Method must be 'uniform' or 'rlss'")






def NysMMDtest(Z, n, m, seed=None, method='uniform', bandwidth=1, k=20):
    """
    Performs the Nyström approximation MMD test.
    Parameters:
    Z (array): Feature matrix of shape (ntot, d).
    n (int): number of data points for first sample
    m (int): number of data points for second sample
    seed (int, optional): Random seed for reproducibility.
    method (str): Nyström sampling method ('uniform' or 'rlss').
    bandwidth (float): Kernel bandwidth parameter.
    k (int): Number of Nyström features.
    Returns:
    MMD (float): value of MMD test statistic
    """
    ntot, d = Z.shape  # Total samples and feature dimension
    assert n + m == ntot, "n + m must be equal to the total size of the dataset"
    
    # Compute Nyström feature mapping
    inds, _ = nys_inds_tf(Z, k, method, bandwidth, seed)  
    psi_Z = nystrom_features_tf(Z, inds, bandwidth)
    
    psi_X = psi_Z[:n]  # First n samples (X group)
    psi_Y = psi_Z[n:]  # Last m samples (Y group)
    
    # Compute mean embeddings
    bar_psi_X = np.mean(psi_X, axis=0)
    bar_psi_Y = np.mean(psi_Y, axis=0)
    
    T = bar_psi_X - bar_psi_Y
    
    # Compute MMD statistic
    MMD = np.sum(T ** 2)  
    
    return MMD
