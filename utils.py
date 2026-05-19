import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruction_errors(model, X, device, batch_size=4096):
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            reconstructed = model(batch)
            batch_errors = torch.mean((reconstructed - batch) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())
    return np.concatenate(errors)


def compute_threshold(errors, labels, num_thresholds: int = 1000):
    """Select threshold that maximizes F1-score on validation data only."""
    thresholds = np.linspace(errors.min(), errors.max(), num_thresholds)
    best_threshold, best_f1 = thresholds[0], -1.0

    for threshold in thresholds:
        y_pred = (errors > threshold).astype(int)
        f1 = f1_score(labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1

    return float(best_threshold), float(best_f1)


def save_threshold(threshold, path):
    np.save(path, np.array([threshold], dtype=np.float32))


def load_threshold(path):
    return float(np.load(path).reshape(-1)[0])
