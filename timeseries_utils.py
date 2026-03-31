import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import sklearn.metrics.pairwise as sp

def euclidean_ts_dist(X1, X2):
    return cdist(X1.reshape(X1.shape[0], -1), 
                 X2.reshape(X2.shape[0], -1), 
                 metric='euclidean')


def dtw_distance(s1, s2, window=None):
    n, m = len(s1), len(s2)
    if window is None:
        window = max(n, m)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i - j) > window:
                continue
            cost = (s1[i-1] - s2[j-1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    
                dtw_matrix[i, j-1],    
                dtw_matrix[i-1, j-1]   
            )
    return np.sqrt(dtw_matrix[n, m])


def pairwise_dtw(X1, X2, window=None):
    n1, n2 = len(X1), len(X2)
    dist_mtx = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist_mtx[i, j] = dtw_distance(X1[i], X2[j], window)
    return dist_mtx


def get_distance_fn(metric='dtw', **kwargs):
    if metric == 'euclidean':
        return euclidean_ts_dist
    elif metric == 'dtw':
        return lambda X1, X2: pairwise_dtw(X1, X2, **kwargs)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def generate_synthetic_ts(n_samples=100, T=50, n_classes=2, 
                          noise_level=0.1, shift_factor=0.5,
                          seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for cls in range(n_classes):
        for _ in range(samples_per_class):
            t = np.linspace(0, 4*np.pi, T)
            
            if cls == 0:
                base = np.sin(t)
            elif cls == 1:
                base = np.cos(t)
            elif cls == 2:
                base = np.sin(2*t)  
            else:
                base = (np.random.randn() * np.sin(t) + 
                       np.random.randn() * np.cos(0.5*t))
            
            phase_shift = np.random.uniform(-shift_factor, shift_factor) * np.pi
            base = np.roll(base, int(phase_shift * T / (2*np.pi)))
            
            noise = np.random.randn(T) * noise_level
            series = base + noise
            
            trend = np.linspace(0, 0.1*cls, T)
            series += trend
            
            X.append(series)
            y.append(cls)
    
    for i in range(n_samples - len(y)):
        cls = np.random.randint(0, n_classes)
        t = np.linspace(0, 4*np.pi, T)
        base = np.sin(t + cls) if cls < 2 else np.cos(t * (cls+1))
        noise = np.random.randn(T) * noise_level
        X.append(base + noise)
        y.append(cls)
    
    return {
        'name': f'synthetic_ts_T{T}_nc{n_classes}',
        'X': np.array(X),  
        'y': np.array(y),
        'metadata': {
            'T': T,
            'n_classes': n_classes,
            'noise_level': noise_level,
            'shift_factor': shift_factor
        }
    }


def generate_arima_like_ts(n_samples=100, T=50, ar_coef=0.8, 
                           ma_coef=0.3, n_classes=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = []
    y = []
    
    for cls in range(n_classes):
        ar = ar_coef * (1 + 0.2*cls)
        ma = ma_coef * (1 - 0.3*cls)
        
        for _ in range(n_samples // n_classes):
            series = np.zeros(T)
            noise = np.random.randn(T) * 0.5
            
            for t in range(1, T):
                ar_part = ar * series[t-1] if t-1 >= 0 else 0
                ma_part = ma * noise[t-1] if t-1 >= 0 else 0
                series[t] = ar_part + ma_part + noise[t]
            
            series = (series - series.mean()) / (series.std() + 1e-8)
            X.append(series)
            y.append(cls)
    
    return {
        'name': f'arima_like_T{T}_nc{n_classes}',
        'X': np.array(X),
        'y': np.array(y),
        'metadata': {'ar_coef': ar_coef, 'ma_coef': ma_coef}
    }

class TSWeightClassifier:
    def __init__(self, base_clf, distance_metric='dtw', 
                 normalize=True, dtw_window=None):
        self.base_clf = base_clf
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.dtw_window = dtw_window
        self._scaler = None
        self._distance_fn = None
        
    def _get_distance_fn(self):
        if self.distance_metric == 'euclidean':
            return euclidean_ts_dist
        elif self.distance_metric == 'dtw':
            return lambda X1, X2: pairwise_dtw(X1, X2, window=self.dtw_window)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")
    
    def _patch_distance_matrix(self):
        self._original_dist_fn = getattr(sp, 'distance_matrix', None)
        sp.distance_matrix = self._distance_fn
    
    def _restore_distance_matrix(self):
        if self._original_dist_fn is not None:
            sp.distance_matrix = self._original_dist_fn
    
    def fit(self, X, y):
        if self.normalize:
            self._scaler = StandardScaler()
            X_norm = np.array([
                self._scaler.fit_transform(x.reshape(-1, 1)).ravel() 
                for x in X
            ])
        else:
            X_norm = X.copy()
        
        self._distance_fn = self._get_distance_fn()
        
        self._patch_distance_matrix()
        try:
            self.base_clf.fit(X_norm, y)
        finally:
            self._restore_distance_matrix()
        
        return self
    
    def predict(self, X):
        if self.normalize:
            X_norm = np.array([
                self._scaler.transform(x.reshape(-1, 1)).ravel() 
                for x in X
            ])
        else:
            X_norm = X.copy()
            
        self._patch_distance_matrix()
        try:
            return self.base_clf.predict(X_norm)
        finally:
            self._restore_distance_matrix()
    
    def predict_proba(self, X):
        if self.normalize:
            X_norm = np.array([
                self._scaler.transform(x.reshape(-1, 1)).ravel() 
                for x in X
            ])
        else:
            X_norm = X.copy()
            
        self._patch_distance_matrix()
        try:
            return self.base_clf.predict_proba(X_norm)
        finally:
            self._restore_distance_matrix()
    
    @property
    def classes_(self):
        return self.base_clf.classes_