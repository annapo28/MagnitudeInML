import numpy as np, pandas as pd, sklearn.datasets, os
from scipy.io import loadmat
from scipy.spatial import distance_matrix
from sklearn.preprocessing import LabelEncoder

ALL_FILES = os.listdir('data') if os.path.exists('data') else []
DATASETS = [f for f in ALL_FILES if f.endswith(('.mat', '.csv'))]

def load_dataset(nm):
    file_path = f'data/{nm}'
    if nm.endswith('.mat'):
        DATA = loadmat(file_path)
        y = np.ravel(DATA['d'], order='C')
        return {'name': nm, 'X': DATA['A'], 'y': y}
    elif nm == 'drug200.csv':
        df = pd.read_csv(file_path)
        df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
        df['BP'] = df['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
        df['Cholesterol'] = df['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})
        X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values.astype(float)
        y = LabelEncoder().fit_transform(df['Drug'].values)
        return {'name': 'drug200', 'X': X, 'y': y}
    elif nm == 'bupa.data':
        df = pd.read_csv(file_path, header=None)
        X = df.iloc[:, :5].values.astype(float)
        y = (df.iloc[:, 5].values.astype(float) > np.median(df.iloc[:, 5].values)).astype(int)
        return {'name': 'liver_disorders', 'X': X, 'y': y}
    elif nm in ['cleveland.data', 'processed.cleveland.data', 'heart_cleveland.csv']:
        df = pd.read_csv(file_path, header=None, na_values='?').dropna()
        X = df.iloc[:, :-1].values.astype(float)
        y = (df.iloc[:, -1].values.astype(int) > 0).astype(int)
        name = 'heart_cleveland_processed' if 'processed' in nm else 'heart_cleveland'
        return {'name': name, 'X': X, 'y': y}
    elif nm in ['eighthr.data', 'ozone_eighthr.csv']:
        df = pd.read_csv(file_path, header=None, na_values='?').dropna()
        X = df.iloc[:, 2:-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
        return {'name': 'ozone_eighthr', 'X': X, 'y': y}
    elif 'letter-recognition' in nm:
        df = pd.read_csv(file_path, header=None)
        X = df.iloc[:, 1:].values.astype(float)
        y = LabelEncoder().fit_transform(df.iloc[:, 0].values)
        return {'name': nm, 'X': X, 'y': y}
    elif nm == 'diabetes.csv':
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        return {'name': 'diabetes.csv', 'X': X, 'y': y}
    elif nm == 'WineQT.csv':
        df = pd.read_csv(file_path)
        if 'Id' in df.columns: df = df.drop('Id', axis=1)
        X = df.drop('quality', axis=1).values.astype(float)
        y = (df['quality'].values >= 6).astype(int)
        return {'name': 'WineQT_binary', 'X': X, 'y': y}
    elif 'wine' in nm.lower() and nm.endswith('.csv'):
        df = pd.read_csv(file_path)
        target_col = next((c for c in ['quality','target','class','Customer_Segment','class_label'] if c in df.columns), df.columns[-1])
        X = df.drop(target_col, axis=1).values.astype(float)
        y_raw = df[target_col].values
        y = LabelEncoder().fit_transform(y_raw) if not np.issubdtype(y_raw.dtype, np.number) else y_raw.astype(int)
        return {'name': nm, 'X': X, 'y': y}
    elif nm.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            target_candidates = ['target','class','label','Class','Target','Label','category','Category','y','Y','result','Result','diagnosis','Diagnosis','type','Type','species','Species']
            target_col = next((c for c in target_candidates if c in df.columns), None)
            if target_col:
                y_raw = df[target_col].values
                X = df.drop(columns=[target_col]).values.astype(float)
            else:
                X = df.iloc[:, :-1].values.astype(float)
                y_raw = df.iloc[:, -1].values
        except:
            df = pd.read_csv(file_path, header=None)
            X = df.iloc[:, :-1].values.astype(float)
            y_raw = df.iloc[:, -1].values
        if np.isnan(X).any():
            col_medians = np.nanmedian(X, axis=0)
            for i in range(X.shape[1]): X[np.isnan(X[:, i]), i] = col_medians[i]
        if y_raw.dtype == 'object' or not np.issubdtype(y_raw.dtype, np.number):
            y = LabelEncoder().fit_transform(y_raw.astype(str))
        else:
            y = y_raw.astype(int)
        if np.isnan(y).any():
            valid = ~np.isnan(y)
            X, y = X[valid], y[valid]
        return {'name': nm, 'X': X, 'y': y}
    else:
        raise ValueError(f"Unsupported file format: {nm}")

def get_checkerboard(low=0, high=4, size=(1000, 2), seed=234):
    np.random.seed(seed)
    X = np.random.uniform(low, high, size=size)
    return {'name': f'{X.shape[1]}-d checkerboard', 'X': X, 'y': np.floor(X).sum(axis=1) % 2}

def get_mnist_small():
    d = sklearn.datasets.load_digits()
    return {'name': 'sklearn digits', 'X': d['data'], 'y': d['target']}

def get_iris():
    d = sklearn.datasets.load_iris()
    return {'name': 'iris', 'X': d['data'], 'y': d['target']}

def add_sklearn_datasets():
    sklearn_sets = []
    for loader, name in [(sklearn.datasets.load_breast_cancer, 'sklearn_breast_cancer'),
                         (sklearn.datasets.load_wine, 'sklearn_wine'),
                         (sklearn.datasets.load_digits, 'sklearn_digits')]:
        try:
            d = loader()
            sklearn_sets.append({'name': name, 'X': d.data, 'y': d.target})
        except: pass
    return sklearn_sets

def remove_close_points(dataset):
    X, y = dataset['X'], dataset['y']
    dist_mtx = distance_matrix(X, X)
    min_dist = dist_mtx[np.where(dist_mtx > 0)].min()
    coords_mean = np.mean(np.abs(X[:, None] - X[None, :]), axis=-1)
    idx = np.where((coords_mean > 0) & (coords_mean < 0.5*min_dist))[0][1:]
    dataset['X'], dataset['y'] = np.delete(X, idx, axis=0), np.delete(y, idx, axis=0)
    return dataset

datasets = []
for nm in DATASETS:
    if nm != 'censusdata.mat':
        try: datasets.append(load_dataset(nm))
        except: continue

datasets = [get_checkerboard(), get_mnist_small(), get_iris()] + add_sklearn_datasets() + datasets
datasets_sorted = sorted(datasets, key=lambda x: (len(np.unique(x['y'])), x['X'].shape[0]))
for i, d in enumerate(datasets_sorted):
    print(f"{i+1:2d}. {d['name']:30s} samples: {d['X'].shape[0]:5d}, features: {d['X'].shape[1]:3d}, classes: {len(np.unique(d['y'])):2d}")