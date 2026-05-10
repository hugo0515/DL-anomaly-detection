from sklearn.metrics import f1_score
import numpy as np


def compute_threshold(errors, labels):
    """
    Compute the optimal anomaly detection threshold by maximizing F1-score
    on the validation set.
    """
    thresholds = np.linspace(errors.min(), errors.max(), 1000)

    best_threshold = thresholds[0]
    best_f1 = 0.0

    for threshold in thresholds:
        y_pred = (errors > threshold).astype(int)
        f1 = f1_score(labels, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def save_threshold(threshold, path="threshold_value.npy"):
    np.save(path, threshold)


def load_threshold(path="threshold_value.npy"):
    return np.load(path)
