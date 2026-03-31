import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_subsample(input_file='data/letter-recognition.data', 
                     output_file='data/letter-recognition-small.data',
                     sample_size=1000, 
                     random_state=42):
    
    df = pd.read_csv(input_file, header=None)
    
    print(f"Original dataset: {len(df)} rows, {df.shape[1]} columns")
    print(f"Creating subsample of {sample_size} rows...")
    
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]   
    X_small, _, y_small, _ = train_test_split(
        X, y, 
        train_size=sample_size,
        stratify=y,
        random_state=random_state
    )
    
    df_small = pd.concat([y_small.reset_index(drop=True), 
                         X_small.reset_index(drop=True)], axis=1)
    
    df_small.to_csv(output_file, header=False, index=False)
    
    print(f"Subsample created: {len(df_small)} rows")
    print(f"Class distribution:")
    print(df_small.iloc[:, 0].value_counts().sort_index())
    
    return df_small

for size in [500, 1000, 2000, 5000]: 
    create_subsample(
        output_file=f'data/letter-recognition-{size}.data',
        sample_size=size
    )