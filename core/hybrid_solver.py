import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from core.magnitude import compute_zeta, magnitude_weights, magnitude_exact


class HybridMagnitudeSolver:
    def __init__(self, t=2.0, threshold=0.1, use_curvature=False, alpha=0.1):
        self.t = t
        self.threshold = threshold
        self.use_curvature = use_curvature
        self.alpha = alpha
        self.local_components = None
        self.block_score_ = None
        self.effective_points_ = None
        self.curvatures_ = None
        self.weights_ = None
        self.magnitude_ = None

    def compute_block_score(self, Z):
        n = Z.shape[0]
        if n == 0:
            return 0.0
        sorted_idx = np.argsort(np.sum(Z, axis=1))
        Z_sorted = Z[np.ix_(sorted_idx, sorted_idx)]
        block_size = max(10, n // 5)
        diagonal_weight = 0.0
        total_weight = np.sum(Z_sorted)
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                if abs(i - j) < block_size * 1.5:
                    block = Z_sorted[
                        i:min(i + block_size, n),
                        j:min(j + block_size, n)
                    ]
                    diagonal_weight += np.sum(block)
        return diagonal_weight / total_weight if total_weight > 0 else 0.0

    def _find_local_components(self, Z):
        if self.threshold is None:
            weights = Z[np.triu_indices(Z.shape[0], k=1)]
            self.threshold = np.percentile(weights, 10)
        adj_matrix = (Z > self.threshold).astype(int)
        n_components, labels = connected_components(
            csgraph=adj_matrix,
            directed=False,
            return_labels=True
        )
        components = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            components[label].append(idx)
        return components, labels

    def _compute_local_curvature(self, X, component_indices):
        if len(component_indices) < 3:
            return 0.0
        X_local = X[component_indices]
        try:
            hull = ConvexHull(X_local)
            volume = hull.volume if X_local.shape[1] > 1 else np.ptp(X_local)
            mag_local = magnitude_exact(X_local, t=self.t)
            curvature = abs(mag_local - volume) / max(volume, 1e-6)
            return min(curvature, 1.0)
        except Exception:
            return 0.0

    def fit(self, X):
        Z = compute_zeta(X, t=self.t)

        self.block_score_ = self.compute_block_score(Z)
        components, _ = self._find_local_components(Z)
        self.local_components = components         
        n = X.shape[0]
        w_final = np.zeros(n)
        self.effective_points_ = []
        self.curvatures_ = []

        for component in self.local_components:
            if len(component) == 0:
                continue

            Z_local = Z[np.ix_(component, component)]
            try:
                w_local = np.linalg.solve(Z_local, np.ones(len(component)))
            except np.linalg.LinAlgError:
                w_local = np.linalg.lstsq(
                    Z_local, np.ones(len(component)), rcond=None
                )[0]

            if self.use_curvature:
                curvature = self._compute_local_curvature(X, component)
                self.curvatures_.append(curvature)
                w_local = w_local * (1.0 + self.alpha * curvature)
            else:
                self.curvatures_.append(0.0)

            for idx, weight in zip(component, w_local):
                w_final[idx] = weight

            centroid = np.mean(X[component], axis=0)
            self.effective_points_.append(centroid)

        if self.use_curvature and len(self.effective_points_) > 1:
            effective_X = np.vstack(self.effective_points_)
            w_global = magnitude_weights(effective_X, t=self.t)
            for comp_idx, component in enumerate(self.local_components):
                if len(component) > 0:
                    local_sum = np.sum(w_final[component])
                    if abs(local_sum) > 1e-12:
                        scale = w_global[comp_idx] / local_sum
                        w_final[component] *= scale

        self.weights_ = w_final
        self.magnitude_ = float(np.sum(w_final))
        return self

    def magnitude(self):
        if self.magnitude_ is None:
            raise RuntimeError("fit(X) перед magnitude()")
        return self.magnitude_

    def get_block_structure_info(self):
        component_sizes = [len(c) for c in self.local_components]
        info = {
            'n_components':   len(self.local_components),
            'component_sizes': component_sizes,
            'threshold_used': self.threshold,
            'block_score':    self.block_score_,
            'use_curvature':  self.use_curvature,
        }
        if self.use_curvature and self.curvatures_ is not None:
            info['curvatures'] = self.curvatures_
        return info