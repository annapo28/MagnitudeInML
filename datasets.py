import numpy as np
import pandas as pd
import sklearn.datasets
from scipy.io import loadmat
from scipy.spatial import distance_matrix
import os
from sklearn.preprocessing import LabelEncoder


# Получаем все файлы из папки data
ALL_FILES = os.listdir('data') if os.path.exists('data') else []

# Фильтруем только нужные файлы
DATASETS = []
for file in ALL_FILES:
    if file.endswith('.mat'):
        DATASETS.append(file)
    elif file.endswith('.csv') or 'letter-recognition' in file:  # Все CSV и letter-recognition файлы
        DATASETS.append(file)

print(f"Found datasets: {DATASETS}")


def load_dataset(nm):
    """Загружает датасет из разных форматов"""
    file_path = f'data/{nm}'
    
    if nm.endswith('.mat'):
        # Загрузка MATLAB .mat файлов
        DATA = loadmat(file_path)
        y = DATA['d']
        y = np.ravel(y, order='C')
        print(f"Loaded .mat: {nm}, X shape: {DATA['A'].shape}, y shape: {y.shape}")
        return {'name': nm, 'X': DATA['A'], 'y': y}
    
    elif 'letter-recognition' in nm:
        # Загрузка letter recognition CSV (без заголовков)
        df = pd.read_csv(file_path, header=None)
        
        # Признаки (столбцы 1-16)
        X = df.iloc[:, 1:].values.astype(float)
        
        # Метки (столбец 0) - буквы A-Z
        y_letters = df.iloc[:, 0].values
        
        # Преобразуем буквы в числа
        le = LabelEncoder()
        y = le.fit_transform(y_letters)
        
        print(f"Loaded CSV: {nm}, X shape: {X.shape}, y shape: {y.shape}")
        print(f"Classes: {len(np.unique(y))} (letters: {sorted(np.unique(y_letters))})")
        return {'name': nm, 'X': X, 'y': y}
    
    elif nm == 'diabetes.csv':
        # Специальная обработка для diabetes.csv (с заголовками)
        df = pd.read_csv(file_path)  # С заголовками
        
        # Последняя колонка 'Outcome' - целевая переменная (0/1)
        # Все остальные колонки - признаки
        X = df.iloc[:, :-1].values.astype(float)  # все кроме последней колонки
        y = df.iloc[:, -1].values  # последняя колонка 'Outcome'
        
        print(f"Loaded diabetes.csv with headers")
        print(f"  Features: {df.columns[:-1].tolist()}")
        print(f"  Target: {df.columns[-1]}")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)} (counts: {np.bincount(y.astype(int))})")
        
        return {'name': 'diabetes.csv', 'X': X, 'y': y}
    
    elif nm == 'WineQT.csv':
        # Специальная обработка для Wine Quality dataset
        df = pd.read_csv(file_path)
        
        # Удаляем Id колонку
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        # 'quality' - целевая переменная (значения 3-9)
        # Преобразуем в бинарную классификацию: хорошее вино (quality >= 6) vs плохое
        X = df.drop('quality', axis=1).values.astype(float)
        y_quality = df['quality'].values
        
        # Бинарная классификация: 1 = хорошее вино (>=6), 0 = плохое
        y = (y_quality >= 6).astype(int)
        
        # Альтернатива: многоклассовая классификация (3 класса)
        # y = np.zeros_like(y_quality)
        # y[y_quality <= 4] = 0  # низкое качество (3-4)
        # y[(y_quality >= 5) & (y_quality <= 6)] = 1  # среднее (5-6)
        # y[y_quality >= 7] = 2  # высокое (7-9)
        
        print(f"Loaded WineQT.csv as binary classification")
        print(f"  Original quality range: {y_quality.min()}-{y_quality.max()}")
        print(f"  Good wine (quality >= 6): {y.sum()} samples ({y.sum()/len(y)*100:.1f}%)")
        print(f"  Bad wine (quality < 6): {len(y)-y.sum()} samples ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        return {'name': 'WineQT_binary', 'X': X, 'y': y}
    
    elif 'wine' in nm.lower() and nm.endswith('.csv'):
        # Универсальная обработка для других wine датасетов
        df = pd.read_csv(file_path)
        
        # Ищем целевую колонку
        target_candidates = ['quality', 'target', 'class', 'Customer_Segment', 'class_label']
        
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            # Берем последнюю колонку
            target_col = df.columns[-1]
        
        X = df.drop(target_col, axis=1).values.astype(float)
        y_raw = df[target_col].values
        
        # Преобразуем в числовые метки если нужно
        if not np.issubdtype(y_raw.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            print(f"  Encoded target labels")
        else:
            y = y_raw.astype(int)
        
        print(f"Loaded wine dataset: {nm}")
        print(f"  Target column: '{target_col}'")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  Classes: {sorted(np.unique(y))} (counts: {np.bincount(y)})")
        
        return {'name': nm, 'X': X, 'y': y}
    
    elif nm.endswith('.csv'):
        # Общая обработка для других CSV файлов
        try:
            # Пробуем с заголовками
            df = pd.read_csv(file_path)
            print(f"Loaded {nm} with headers")
            
            # Пытаемся определить целевую переменную
            # Расширенный список возможных названий целевых колонок
            target_candidates = ['target', 'class', 'label', 'Class', 'Target', 
                               'Label', 'category', 'Category', 'y', 'Y',
                               'result', 'Result', 'diagnosis', 'Diagnosis',
                               'type', 'Type', 'species', 'Species']
            
            target_col = None
            for col in target_candidates:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                # Используем найденную колонку как целевую
                y_raw = df[target_col].values
                X = df.drop(columns=[target_col]).values.astype(float)
                print(f"  Using target column: '{target_col}'")
            else:
                # Берем последнюю колонку
                X = df.iloc[:, :-1].values.astype(float)
                y_raw = df.iloc[:, -1].values
                print(f"  Using last column as target")
                
        except Exception as e:
            # Без заголовков
            df = pd.read_csv(file_path, header=None)
            print(f"Loaded {nm} without headers")
            X = df.iloc[:, :-1].values.astype(float)
            y_raw = df.iloc[:, -1].values
        
        # Обработка пропущенных значений в X
        if np.isnan(X).any():
            print(f"  Found {np.isnan(X).sum()} missing values in features, filling with column median")
            col_medians = np.nanmedian(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_medians[i]
        
        # Кодируем метки если они не числовые
        if y_raw.dtype == 'object' or not np.issubdtype(y_raw.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            print(f"  Encoded target labels to numeric")
        else:
            y = y_raw.astype(int)
        
        # Проверяем на пропущенные значения в y
        if np.isnan(y).any():
            print(f"  Warning: Found {np.isnan(y).sum()} NaN values in target, removing those samples")
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
        
        print(f"Loaded CSV: {nm}, X shape: {X.shape}, y shape: {y.shape}")
        print(f"Classes: {len(np.unique(y))} (values: {sorted(np.unique(y))})")
        if len(np.unique(y)) <= 10:
            print(f"  Class distribution: {np.bincount(y)}")
        
        return {'name': nm, 'X': X, 'y': y}
    
    else:
        raise ValueError(f"Unsupported file format: {nm}")


def get_checkerboard(low=0, high=4, size=(1000, 2), seed=234):
    np.random.seed(seed)
    X = np.random.uniform(low, high, size=size)
    y = np.floor(X).sum(axis=1) % 2
    return {
        'name': f'{X.shape[1]}-d checkerboard',
        'X': X, 'y': y,
    }


def get_mnist_small():
    d = sklearn.datasets.load_digits()
    return {
        'name': 'sklearn digits',
        'X': d['data'],
        'y': d['target'],
    }


def get_iris():
    d = sklearn.datasets.load_iris()
    return {'name': 'iris', 'X': d['data'], 'y': d['target']}


# Функция для добавления датасетов из sklearn
def add_sklearn_datasets():
    """Добавляет популярные датасеты из sklearn"""
    sklearn_sets = []
    
    # Breast Cancer Wisconsin (2 класса)
    try:
        bc = sklearn.datasets.load_breast_cancer()
        sklearn_sets.append({
            'name': 'sklearn_breast_cancer',
            'X': bc.data,
            'y': bc.target
        })
    except Exception as e:
        print(f"Error loading breast_cancer: {e}")
    
    # Wine (3 класса)
    try:
        wine = sklearn.datasets.load_wine()
        sklearn_sets.append({
            'name': 'sklearn_wine',
            'X': wine.data,
            'y': wine.target
        })
    except Exception as e:
        print(f"Error loading wine: {e}")
    
    # Digits (10 классов) - уже есть как get_mnist_small, но добавим для полноты
    try:
        digits = sklearn.datasets.load_digits()
        sklearn_sets.append({
            'name': 'sklearn_digits',
            'X': digits.data,
            'y': digits.target
        })
    except Exception as e:
        print(f"Error loading digits: {e}")
    
    return sklearn_sets


def remove_close_points(dataset):
    # if points are too close in each coordinate they can cause problems
    # e.g. in iris, there are two points with coordinates
    # [6.4, 2.8, 5.6, 2.1]
    # [6.4, 2.8, 5.6, 2.2]
    # which caused magintude to behave poorly
    X = dataset['X']
    y = dataset['y']

    dist_mtx = distance_matrix(X, X)
    min_dist = dist_mtx[np.where(dist_mtx > 0)].min()

    coords_mean = np.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(coords_mean.shape[0]):
        for j in range(coords_mean.shape[1]):
            coords_mean[i, j] = np.mean(np.abs(X[i] - X[j]))

    indexes = np.where((coords_mean > 0) & (coords_mean < 0.5*min_dist))[0]
    indexes_to_remove = indexes[1:]
    X = np.delete(X, indexes_to_remove, axis=0)
    y = np.delete(y, indexes_to_remove, axis=0)

    dataset['X'] = X
    dataset['y'] = y

    return dataset


# Загружаем все датасеты
datasets = []
for nm in DATASETS:
    if nm != 'censusdata.mat':  # Пропускаем censusdata.mat как в оригинале
        try:
            dataset = load_dataset(nm)
            datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {nm}: {e}")
            continue

# Добавляем синтетические датасеты
other_datasets = [get_checkerboard(), get_mnist_small(), get_iris()]

# Добавляем датасеты из sklearn
sklearn_datasets = add_sklearn_datasets()

# Объединяем все датасеты
datasets = other_datasets + sklearn_datasets + datasets

print(f"\nTotal loaded datasets: {len(datasets)}")
print("=" * 70)

# Сортируем по количеству классов для удобства анализа
datasets_sorted = sorted(datasets, key=lambda x: (len(np.unique(x['y'])), x['X'].shape[0]))

for i, d in enumerate(datasets_sorted):
    n_classes = len(np.unique(d['y']))
    n_samples, n_features = d['X'].shape
    print(f"{i+1:2d}. {d['name']:30s} "
          f"samples: {n_samples:5d}, features: {n_features:3d}, classes: {n_classes:2d}")

print("\nSummary by number of classes:")
class_counts = {}
for d in datasets_sorted:
    n_classes = len(np.unique(d['y']))
    if n_classes not in class_counts:
        class_counts[n_classes] = 0
    class_counts[n_classes] += 1

for n_classes in sorted(class_counts.keys()):
    print(f"  {n_classes} classes: {class_counts[n_classes]} dataset(s)")