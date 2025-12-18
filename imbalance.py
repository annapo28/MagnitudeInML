import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from magloss import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import csv
import os


def set_seed(seed: int = 2048):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_classification(majority=0.5, nclasses=10, nfeatures=20, nsamples=10000, random_state=42, nredundant=5):
    minority = (1 - majority) / (nclasses - 1)
    X, y = make_classification(
        n_samples=nsamples,
        n_features=nfeatures,
        n_informative=nfeatures - nredundant,
        n_redundant=nredundant,
        n_classes=nclasses,
        n_clusters_per_class=1,
        weights=[majority] + [minority] * (nclasses - 1),
        flip_y=0.01,
        random_state=random_state
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    y_train_oh = torch.nn.functional.one_hot(y_train_t, num_classes=nclasses).float()
    y_test_oh = torch.nn.functional.one_hot(y_test_t, num_classes=nclasses).float()

    return X_train_t, X_test_t, y_train_t, y_test_t, y_train_oh, y_test_oh

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)  # logits output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # no softmax here
        return x


def train_and_eval(X_train, y_train, X_test, y_test, loss_fn, one_hot=False, from_logits=True,
                   num_classes=10, epochs=100, batch_size=32, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(X_train.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    auc_pr = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    f1_micro = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='micro')
    f1_macro = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')

    train_data = TensorDataset(X_train, y_train if not one_hot else y_train.argmax(dim=1))
    test_data = TensorDataset(X_test, y_test if not one_hot else y_test.argmax(dim=1))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    history = {
        "val_accuracy": [], "val_loss": [], "val_auc": [],
        "val_micro": [], "val_macro": [],
        "val_cce": [], "val_mse": []
    }

    cce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()

    start_time = time.time()

    for epoch in tqdm(range(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)

            if one_hot:
                if not from_logits:
                    probs = torch.softmax(logits, dim=1)
                    loss = loss_fn(probs, torch.nn.functional.one_hot(yb, num_classes=num_classes).float())
                else:
                    loss = loss_fn(logits, torch.nn.functional.one_hot(yb, num_classes=num_classes).float())
            else:
                loss = loss_fn(logits, yb)

            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        val_loss, val_acc = 0, 0
        val_cce, val_mse = 0, 0
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

                if one_hot:
                    if not from_logits:
                        probs = torch.softmax(logits, dim=1)
                        loss = loss_fn(probs, torch.nn.functional.one_hot(yb, num_classes=num_classes).float())
                    else:
                        loss = loss_fn(logits, torch.nn.functional.one_hot(yb, num_classes=num_classes).float())
                else:
                    loss = loss_fn(logits, yb)

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                # Extra metrics
                val_cce += cce_loss_fn(logits, yb).item()
                val_mse += mse_loss_fn(probs, torch.nn.functional.one_hot(yb, num_classes=num_classes).float()).item()

                val_acc += (preds == yb).sum().item()
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        history["val_loss"].append(val_loss / len(test_loader))
        history["val_accuracy"].append(val_acc / len(test_loader.dataset))
        history["val_auc"].append(auc_pr(all_probs, all_targets).item())
        history["val_micro"].append(f1_micro(all_preds, all_targets).item())
        history["val_macro"].append(f1_macro(all_preds, all_targets).item())
        history["val_cce"].append(val_cce / len(test_loader))
        history["val_mse"].append(val_mse / len(test_loader))

    elapsed_time = time.time() - start_time

    # Find best values + epochs
    best_results = {}
    for key in history:
        if "loss" in key or "cce" in key or "mse" in key:
            best_val = min(history[key])
        else:
            best_val = max(history[key])
        best_epoch = history[key].index(best_val) + 1
        best_results[key] = (best_val, best_epoch)

    print("\n=== Best Results ===")
    for k, (val, epoch) in best_results.items():
        print(f"{k:>12}: {val:.4f} (Epoch {epoch})")

    print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    return history

def plot_results(results_dict, majority):
    maj_name = str(int(majority*100))
    metrics = ["val_accuracy", "val_loss", "val_auc", "val_micro", "val_macro", "val_cce", "val_mse"]
    titles = {
        "val_accuracy": "Validation Accuracy Progression",
        "val_loss": "Validation Loss Progression",
        "val_auc": "Validation AUC (PR) Progression",
        "val_micro": "Validation F1-Score (Micro) Progression",
        "val_macro": "Validation F1-Score (Macro) Progression",
        "val_cce": "Validation CCE Progression",
        "val_mse": "Validation MSE Progression",
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for name, hist in results_dict.items():
            plt.plot(hist[metric], label=name)
        plt.title(titles[metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric.replace("val_", "").capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric}_comparison_imbalance_{maj_name}.png")
        plt.show()



def save_history_to_csv(history, filename):
    """Save training history (dict of lists) to CSV."""
    keys = list(history.keys())
    epochs = list(range(1, len(history[keys[0]]) + 1))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(epochs)):
            row = [epochs[i]] + [history[k][i] for k in keys]
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducability script for CardLoss paper.")
    
    # get_classification args
    parser.add_argument("--majority", type=float, default=0.5, help="Majority class proportion.")
    parser.add_argument("--nclasses", type=int, default=10, help="Number of classes.")
    parser.add_argument("--nfeatures", type=int, default=20, help="Number of features.")
    parser.add_argument("--nsamples", type=int, default=10000, help="Number of samples.")
    parser.add_argument("--random_state", type=int, default=2048, help="Random seed.")
    parser.add_argument("--nredundant", type=int, default=5, help="Number of redundant features.")

    # train_and_eval args
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")

    args = parser.parse_args()

    set_seed(args.random_state)

    X_train, X_test, y_train, y_test, y_train_oh, y_test_oh = get_classification(
        majority=args.majority,
        nclasses=args.nclasses,
        nfeatures=args.nfeatures,
        nsamples=args.nsamples,
        random_state=args.random_state,
        nredundant=args.nredundant
    )

    loss_functions = {
        "Magnitude": (mag_loss(), True, False),
        "Spread": (spread_loss(), True, False),
        "Categorical Crossentropy": (nn.CrossEntropyLoss(), False, True),
        "MSE (One-hot)": (nn.MSELoss(), True, False),
    }

    results = {}
    for name, (loss_fn, one_hot, from_logits) in loss_functions.items():
        print(f"\nTraining with {name}...")
        hist = train_and_eval(
            X_train, y_train_oh if one_hot else y_train,
            X_test, y_test_oh if one_hot else y_test,
            loss_fn, one_hot=one_hot, from_logits=from_logits,
            num_classes=args.nclasses,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        # Store in results dict
        results[name] = hist

        maj_name = str(int(args.majority*100))
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        csv_filename = f"results_{safe_name}_{maj_name}.csv"
        save_history_to_csv(hist, csv_filename)
        print(f"Saved CSV to {csv_filename}")

    plot_results(results, args.majority)