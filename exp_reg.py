import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from magnitude_loss import MagnitudeRegressionLoss

class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def create_outlier_dataset(n_samples=1000, n_features=20, outlier_frac=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                           noise=10, random_state=42)
    y = y.reshape(-1, 1)

    n_outliers = int(n_samples * outlier_frac)
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_idx] += np.random.uniform(50, 100, size=n_outliers).reshape(-1, 1)
    
    return X, y

def train_model(model, loader, loss_fn, optimizer, epochs=50):
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(loader))
    return history

if __name__ == "__main__":
    print("=== Тестирование Magnitude Loss для регрессии ===\n")
    
    X, y = create_outlier_dataset(n_samples=2000, outlier_frac=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    to_tensor = lambda x: torch.FloatTensor(x)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(to_tensor(X_train), to_tensor(y_train)),
        batch_size=64, shuffle=True
    )
    
    results = {}
    
    print("Обучение с MSE")
    model_mse = RegressionNet(X_train.shape[1])
    opt_mse = optim.Adam(model_mse.parameters(), lr=0.01)
    loss_mse = nn.MSELoss()
    train_model(model_mse, train_loader, loss_mse, opt_mse, epochs=50)
    
    print("Обучение с MAE")
    model_mae = RegressionNet(X_train.shape[1])
    opt_mae = optim.Adam(model_mae.parameters(), lr=0.01)
    loss_mae = nn.L1Loss()
    train_model(model_mae, train_loader, loss_mae, opt_mae, epochs=50)
     
    print("Обучение с Magnitude Loss (t=1.0)...")
    model_mag = RegressionNet(X_train.shape[1])
    opt_mag = optim.Adam(model_mag.parameters(), lr=0.01)
    loss_mag = MagnitudeRegressionLoss(scale=1.0)
    train_model(model_mag, train_loader, loss_mag, opt_mag, epochs=50)
    
    # Оценка качества (на тесте, MAE для робастности)
    def evaluate(model, X, y):
        model.eval()
        with torch.no_grad():
            pred = model(to_tensor(X))
            mae = torch.mean(torch.abs(pred - to_tensor(y))).item()
            mse = torch.mean((pred - to_tensor(y))**2).item()
        return mae, mse
    
    mae_mse, mse_mse = evaluate(model_mse, X_test, y_test)
    mae_mae, mse_mae = evaluate(model_mae, X_test, y_test)
    mae_mag, mse_mag = evaluate(model_mag, X_test, y_test)
    
    print(f"{'Метод':<20} | {'Test MAE':<10} | {'Test MSE':<10}")
    print("-" * 45)
    print(f"{'MSE Loss':<20} | {mae_mse:<10.4f} | {mse_mse:<10.4f}")
    print(f"{'MAE Loss':<20} | {mae_mae:<10.4f} | {mse_mae:<10.4f}")
    print(f"{'Magnitude Loss':<20} | {mae_mag:<10.4f} | {mse_mag:<10.4f}")
