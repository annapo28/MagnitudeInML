import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular

def compute_zeta(X, t=1.0, metric='euclidean'):
    dist_matrix = cdist(X, X, metric=metric) * t
    return np.exp(-dist_matrix)

def magnitude_exact(X, t=1.0):
    Z = compute_zeta(X, t)
    try:
        L = cholesky(Z, lower=True)
        y = solve_triangular(L, np.ones(Z.shape[0]), lower=True)
        x = solve_triangular(L.T, y, lower=False)
        return np.sum(x)
    except np.linalg.LinAlgError:
        return np.sum(np.linalg.pinv(Z) @ np.ones(Z.shape[0]))

def magnitude_weights(X, t=1.0):
    Z = compute_zeta(X, t)
    try:
        L = cholesky(Z, lower=True)
        y = solve_triangular(L, np.ones(Z.shape[0]), lower=True)
        w = solve_triangular(L.T, y, lower=False)
        return w
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Z, np.ones(Z.shape[0]), rcond=None)[0]