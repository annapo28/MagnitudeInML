import numpy as np
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


def schur_comp(Z, B, C, D):
    """Вычисление дополнения Шура с обработкой разных размерностей"""
    result = D - C.dot(Z).dot(B)
    # Гарантируем, что результат - скаляр или 1D массив
    if hasattr(result, 'ndim') and result.ndim > 0:
        result = result.ravel()
    return result


def cdf(pt, dist):
    """Cumulative Distribution Function - доля точек с весом меньше pt"""
    return ((dist < pt).sum()) / dist.shape[0]


def abs_val(pt, dist):
    """Абсолютное значение (переименовано, чтобы не конфликтовать с built-in abs)"""
    return float(np.abs(pt))


class WeightClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, wt_fn=cdf, magn_scale=None, class_ts=None):
        super().__init__()
        self.wt_fn = wt_fn
        self.magn_scale = magn_scale
        self.class_ts = class_ts
        self._classes = None
        self._info = None
        
    def _setup_classes(self, y):
        _classes = np.unique(y)
        _classes.sort()
        self._classes = _classes
        self.classes_ = _classes  # Для совместимости со sklearn
        
        if self.class_ts is None:
            self.class_ts = np.ones(shape=self._classes.shape, dtype='float')
            err_msg = 'class_ts.shape does not match _classes.shape'
            assert self.class_ts.shape == self._classes.shape, err_msg

    def _setup_info(self, X, y):
        if self.magn_scale is None:
            self._info = {}
            for c in self._classes:
                d = {}
                d['X'] = X[y == c]
                class_index = np.argwhere(self._classes == c)[0][0]
                class_t = self.class_ts[class_index]
                dist_mtx = distance_matrix(d['X'], d['X'])
                
                # Регуляризация для стабильности
                epsilon = 1e-8 * np.eye(dist_mtx.shape[0])
                
                if dist_mtx.shape[0] >= 1000:
                    inv_fn = np.linalg.pinv
                else:
                    inv_fn = np.linalg.inv
                    
                try:
                    # Добавляем небольшую регуляризацию
                    similarity = np.exp(-class_t * dist_mtx) + epsilon
                    d['Z'] = inv_fn(similarity)
                except Exception as e:
                    print(f'Exception {e} for class {c} t value {class_t}')
                    # Больше регуляризации при проблемах
                    D = np.exp(-class_t * dist_mtx) + 0.1 * np.identity(n=dist_mtx.shape[0])
                    d['Z'] = inv_fn(D)
                    
                d['wts'] = d['Z'].sum(axis=1)
                d['t'] = class_t
                self._info[c] = d
        else:
            self._info = {}
            for c in self._classes:
                d = {}
                d['X'] = X[y == c]
                dist_mtx = distance_matrix(d['X'], d['X'])
                
                if dist_mtx.shape[0] >= 1000:
                    inv_fn = np.linalg.pinv
                else:
                    inv_fn = np.linalg.inv
                    
                ts = np.linspace(0.1, 10., 30)
                Zs = []
                for t in ts:
                    try:
                        similarity = np.exp(-t * dist_mtx) + 1e-8 * np.eye(dist_mtx.shape[0])
                        Z = inv_fn(similarity)
                        Zs.append(Z)
                    except Exception as e:
                        print(f'Exception: {e} for t: {t} perturbing matrix')
                        D = np.exp(-t * dist_mtx) + 0.1 * np.identity(n=dist_mtx.shape[0])
                        Z = inv_fn(D)
                        Zs.append(Z)

                magnitudes = np.array([Z.sum() for Z in Zs])
                index = np.argmin(np.abs(magnitudes - self.magn_scale))
                t = ts[index]
                Zt = Zs[index]
                wts = Zt.sum(axis=1)

                d['ts'] = ts
                d['Zs'] = Zs
                d['magnitudes'] = magnitudes
                d['t'] = t
                d['Z'] = Zt
                d['wts'] = wts
                self._info[c] = d

    def fit(self, X, y):
        """Обучение классификатора"""
        self._setup_classes(y)
        self._setup_info(X, y)
        return self

    def _compute_weight(self, C, Z, schur):
        """Вычисление веса с обработкой разных размерностей schur"""
        # Преобразуем schur в скаляр
        if hasattr(schur, 'ndim') and schur.ndim > 0:
            if schur.size == 1:
                schur_scalar = float(schur[0])
            else:
                # Берем первый элемент или среднее, если массив
                schur_scalar = float(schur[0])
        else:
            schur_scalar = float(schur)
            
        # Защита от деления на ноль
        if np.abs(schur_scalar) < 1e-12:
            schur_scalar = 1e-12 if schur_scalar >= 0 else -1e-12
            
        # Вычисление веса
        try:
            term1 = (-1.0 / schur_scalar) * C.dot(Z).sum()
            wt = term1 + (1.0 / schur_scalar)
        except Exception as e:
            # Резервный вариант при ошибках
            wt = 0.0
            
        return wt

    def predict(self, new_points):
        """Предсказание классов для новых точек"""
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            Z = self._info[cls]['Z']
            wts = self._info[cls]['wts']
            t = self._info[cls]['t']
            Cs = np.exp(-t * distance_matrix(new_points, X))
            pred = []
            
            for c_i in Cs:
                C = c_i[np.newaxis]  # shape (1, n)
                B = C.T              # shape (n, 1)
                schur = schur_comp(Z, B, C, 1.0)
                wt = self._compute_weight(C, Z, schur)
                pred.append(self.wt_fn(wt, wts))
                
            res.append(np.array(pred))
            
        preds = np.vstack(res)
        preds = np.argmin(preds, axis=0)
        pred_class = np.array([self._classes[_] for _ in preds])
        return pred_class

    def predict_proba(self, new_points):
        """Вероятности принадлежности к классам"""
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            Z = self._info[cls]['Z']
            wts = self._info[cls]['wts']
            t = self._info[cls]['t']
            Cs = np.exp(-t * distance_matrix(new_points, X))
            pred = []
            
            for c_i in Cs:
                C = c_i[np.newaxis]
                B = C.T
                schur = schur_comp(Z, B, C, 1.0)
                wt = self._compute_weight(C, Z, schur)
                pred.append(self.wt_fn(wt, wts))
                
            res.append(np.array(pred))
            
        preds = np.vstack(res)
        # Преобразуем в вероятности (softmax)
        preds = np.exp(-preds)  # Меньшие значения = большая уверенность
        preds = preds / preds.sum(axis=0, keepdims=True)
        return preds.T


class WeightClassifierCDF(WeightClassifier):
    """Версия с CDF функцией"""
    def __init__(self, magn_scale=None, class_ts=None):
        super().__init__(wt_fn=cdf, magn_scale=magn_scale, class_ts=class_ts)


class WeightClassifierABS(WeightClassifier):
    """Версия с абсолютным значением"""
    def __init__(self, magn_scale=None, class_ts=None):
        super().__init__(wt_fn=abs_val, magn_scale=magn_scale, class_ts=class_ts)


class PowerClassifier(WeightClassifier):
    """Версия с интегрированием по всем t"""
    def __init__(self, wt_fn=cdf, tol=1e-3, t_max=10.):
        self.tol = tol
        self.t_max = t_max
        super().__init__(wt_fn)

    def _setup_info(self, X, y):
        self._info = {}
        for c in self._classes:
            d = {}
            d['X'] = X[y == c]
            dist_mtx = distance_matrix(d['X'], d['X'])
            d['ts'] = np.linspace(self.tol, self.t_max, 20)
            
            if dist_mtx.shape[0] >= 10000:
                inv_fn = np.linalg.pinv
            else:
                inv_fn = np.linalg.inv
                
            # Добавляем регуляризацию
            Zs = []
            for t in d['ts']:
                similarity = np.exp(-t * dist_mtx) + 1e-8 * np.eye(dist_mtx.shape[0])
                Zs.append(inv_fn(similarity))
                
            d['Zs'] = Zs
            wts = [np.exp(-t) * Z.sum(axis=1) for t, Z in zip(d['ts'], d['Zs'])]
            wt_mtx = np.vstack(wts)
            d['powers'] = wt_mtx.sum(axis=0) * (d['ts'][1] - d['ts'][0])
            self._info[c] = d

    def predict(self, new_points):
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            ts = self._info[cls]['ts']
            Zs = self._info[cls]['Zs']
            powers = self._info[cls]['powers']
            Cs = np.exp(-distance_matrix(new_points, X))
            pred = []
            
            for c_i in Cs:
                power = 0
                C = c_i[np.newaxis]
                B = C.T
                
                for Z, t in zip(Zs, ts):
                    schur = schur_comp(Z, B, C, 1.0)
                    wt = self._compute_weight(C, Z, schur)
                    power += (wt * np.exp(-t)) * (ts[1] - ts[0])
                    
                pred.append(self.wt_fn(power, powers))
                
            res.append(np.array(pred))
            
        preds = np.vstack(res)
        preds = np.argmin(preds, axis=0)
        pred_class = np.array([self._classes[_] for _ in preds])
        return pred_class


class PowerClassifierCDF(PowerClassifier):
    def __init__(self, tol=1e-3):
        super().__init__(cdf, tol)


class PowerClassifierABS(PowerClassifier):
    def __init__(self, tol=1e-3):
        super().__init__(abs_val, tol)

TSNE_Classifier = WeightClassifierCDF