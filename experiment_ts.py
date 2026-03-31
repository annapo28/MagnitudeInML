import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from timeseries_utils import (
    generate_synthetic_ts, 
    generate_arima_like_ts,
    TSWeightClassifier
)
import Classifier as classifier  

def run_magnitude_ts_experiment(t_values, n_datasets=5, 
                                T=30, n_samples=60, seed=42):
    results = []
    
    for exp_idx in range(n_datasets):
        dataset = generate_synthetic_ts(
            n_samples=n_samples, 
            T=T, 
            n_classes=2,
            noise_level=0.1 + 0.05*exp_idx,
            shift_factor=0.3 + 0.1*exp_idx,
            seed=seed + exp_idx
        )
        
        X, y = dataset['X'], dataset['y']

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, train_size=0.7, stratify=y, random_state=seed
        )
        
        for magn_scale in t_values:
            for clf_name, clf in [
                ('WeightCDF', classifier.WeightClassifierCDF(magn_scale=magn_scale)),
                ('WeightABS', classifier.WeightClassifierABS(magn_scale=magn_scale)),
            ]:
                ts_clf = TSWeightClassifier(
                    base_clf=clf,
                    distance_metric='dtw',
                    normalize=True,
                    dtw_window=10 
                )
                
                start = time.time()
                ts_clf.fit(train_X, train_y)
                fit_time = time.time() - start
                
                start = time.time()
                preds = ts_clf.predict(test_X)
                pred_time = time.time() - start
                
                acc = (preds == test_y).mean()
                
                results.append({
                    'dataset_idx': exp_idx,
                    'classifier': clf_name,
                    'magn_scale': magn_scale,
                    'accuracy': acc,
                    'fit_time': fit_time,
                    'predict_time': pred_time,
                    'total_time': fit_time + pred_time,
                    'T': T,
                    'n_samples': n_samples,
                    'noise_level': dataset['metadata']['noise_level']
                })
                print(f"[{exp_idx+1}/{n_datasets}] {clf_name} | "
                      f"magn={magn_scale:.2f} | acc={acc:.3f} | "
                      f"time={fit_time+pred_time:.2f}s")
    
    return pd.DataFrame(results)


def plot_magnitude_effect(results_df):
    
    plt.figure(figsize=(10, 6))
    
    agg = results_df.groupby(['classifier', 'magn_scale']).agg({
        'accuracy': ['mean', 'std'],
        'total_time': 'mean'
    }).round(3)
    
    for clf in results_df['classifier'].unique():
        subset = agg.loc[clf] if clf in agg.index.get_level_values(0) else None
        if subset is not None:
            plt.errorbar(
                subset.index, 
                subset[('accuracy', 'mean')],
                yerr=subset[('accuracy', 'std')],
                label=clf,
                marker='o'
            )
    
    plt.xlabel('magn_scale (t)')
    plt.ylabel('Accuracy (mean ± std)')
    plt.title('Влияние магнитуды на качество классификации временных рядов')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ts_magnitude_effect.png', dpi=150)
    plt.show()
    
    plt.figure(figsize=(10, 4))
    for clf in results_df['classifier'].unique():
        subset = results_df[results_df['classifier'] == clf]
        plt.plot(subset['magn_scale'], subset['total_time'], 
                'o-', label=clf, alpha=0.7)
    
    plt.xlabel('magn_scale')
    plt.ylabel('Total time (s)')
    plt.title('Время работы vs магнитуда')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ts_time_vs_magnitude.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    t_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    
    print("Начало")
    results = run_magnitude_ts_experiment(
        t_values=t_values,
        n_datasets=3, 
        T=40,
        n_samples=50
    )
    
    results.to_csv('ts_magnitude_results.csv', index=False)
    print(f"\nРезультаты сохранены: {len(results)} записей")
    
    print("Строим графики")
    plot_magnitude_effect(results)
    
    summary = results.groupby(['classifier', 'magn_scale'])['accuracy'].agg(['mean', 'std']).round(3)
    print(summary)