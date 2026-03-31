import Classifier as classifier
import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import time
from tqdm import tqdm

clfs = [
    classifier.WeightClassifierCDF(),
    classifier.WeightClassifierABS(),
    KNeighborsClassifier(),
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]

pipelines = [
    make_pipeline(StandardScaler(), clf)
    for clf in clfs
]

results = pd.DataFrame(
    columns=[
        'experiment_number',
        'dataset_name',
        'classifier_name',
        'accuracy',
        'confusion_matrix',
        'time (s)',
    ]
)

for experiment_iter in range(10):
    filtered_datasets = [
        data
        for data
        in datasets.datasets[:]
        if data['name'] != 'mushdata.mat'
    ]
    
    for d in tqdm(filtered_datasets, desc="Datasets"):
        X = d['X']
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            d['y'],
            train_size=0.7,
        )
        for pipeline in pipelines:
            pipeline[-1].class_ts = None
            try:
                start = time.time()
                pipeline.fit(train_X, train_y)
                predicted = pipeline.predict(test_X)
                end = time.time()
                
                new_row = pd.DataFrame({
                    'experiment_number': [experiment_iter],
                    'dataset_name': [d['name']],
                    'classifier_name': [type(pipeline[-1]).__name__],
                    'accuracy': [(predicted == test_y).sum() / test_y.shape[0]],
                    'confusion_matrix': [[confusion_matrix(test_y, predicted)]],
                    'time (s)': [end-start]
                })
                results = pd.concat([results, new_row], ignore_index=True)
                
            except Exception as e:
                print(f'Exception: {e}')
                new_row = pd.DataFrame({
                    'experiment_number': [experiment_iter],
                    'dataset_name': [d['name']],
                    'classifier_name': [type(pipeline[-1]).__name__],
                    'accuracy': [None],
                    'confusion_matrix': [None],
                    'time (s)': [0.]
                })
                results = pd.concat([results, new_row], ignore_index=True)

table = results.drop(
    [
        'confusion_matrix',
        'experiment_number',
        'time (s)'
    ],
    axis=1
).groupby(
    [
        'dataset_name',
        'classifier_name',
    ]
).agg(lambda x: f'{np.mean(x):.2f} \u00B1 {np.std(x):.2f}').unstack().copy()

print(table.accuracy.drop('WeightClassifierCDF', axis=1).rename(
    columns={
        'KNeighborsClassifier': 'K-Neighbors',
        'LogisticRegression': 'Logistic Reg.',
        'RandomForestClassifier': 'Random Forest',
        'SVC': 'SVM',
        'WeightClassifierABS': 'Weight',
    }
).drop(
    ['checkdata.mat', 'bupadata.mat', 'pimadata.mat', 'wpbc60data.mat'],
    axis=0
).to_latex())

time_table = results.drop(
    [
        'confusion_matrix',
        'experiment_number',
        'accuracy'
    ],
    axis=1
).groupby(
    [
        'dataset_name',
        'classifier_name',
    ]
).agg(lambda x: f'{np.mean(x):.2f} \u00B1 {np.std(x):.2f}').unstack().copy()

print(time_table['time (s)'].drop('WeightClassifierCDF', axis=1).rename(
    columns={
        'KNeighborsClassifier': 'K-Neighbors',
        'LogisticRegression': 'Logistic Reg.',
        'RandomForestClassifier': 'Random Forest',
        'SVC': 'SVM',
        'WeightClassifierABS': 'Weight',
    }
).drop(
    ['checkdata.mat', 'bupadata.mat', 'pimadata.mat', 'wpbc60data.mat'],
    axis=0
).to_latex())

t_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

t_results = pd.DataFrame(
    columns=[
        'dataset_name',
        'classifier',
        't',
        'fit_time',
        'predict_time',
        'total_time'
    ]
)

for d in datasets.datasets[:3]:
    X = d['X']
    y = d['y']

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    for t in t_values:
        for clf in [
            classifier.WeightClassifierCDF(class_ts=None),
            classifier.WeightClassifierABS(class_ts=None)
        ]:
            n_classes = len(np.unique(train_y))
            clf.class_ts = np.full(n_classes, t)

            start_fit = time.time()
            clf.fit(train_X, train_y)
            end_fit = time.time()

            start_pred = time.time()
            clf.predict(test_X)
            end_pred = time.time()

            t_results = pd.concat([
                t_results,
                pd.DataFrame({
                    'dataset_name': [d['name']],
                    'classifier': [type(clf).__name__],
                    't': [t],
                    'fit_time': [end_fit - start_fit],
                    'predict_time': [end_pred - start_pred],
                    'total_time': [(end_fit - start_fit) + (end_pred - start_pred)]
                })
            ], ignore_index=True)

time_vs_t = (
    t_results
    .groupby(['classifier', 'dataset_name', 't'])
    .agg('mean')
    .reset_index()
)

print("\nВлияние масштаба t на время работы WeightClassifier")
print(time_vs_t.pivot_table(
    index='t', columns=['classifier', 'dataset_name'], values='total_time'
).round(3))

diag_results = []

for d in datasets.datasets[:3]:
    X = d['X']
    y = d['y']

    for cls in np.unique(y):
        Xc = X[y == cls]
        n = Xc.shape[0]
        if n < 2:
            continue

        dist_mtx = np.linalg.norm(Xc[:, None] - Xc[None, :], axis=2)
        min_dist = np.min(dist_mtx[np.where(dist_mtx > 0)])

        t_diag_dom = np.log(n-1) / min_dist

        diag_results.append({
            'dataset_name': d['name'],
            'class': cls,
            'n_points': n,
            'min_dist': min_dist,
            't_diag_dom': t_diag_dom
        })

diag_results_df = pd.DataFrame(diag_results)
print("\nПорог диагональной доминантности для каждого класса")
print(diag_results_df.round(3))
