import numpy as np

def compute_threshold(errors, factor=1.5):
    """
    Compute the anomaly detection threshold using the interquartile range (IQR) method.
    """
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr = q3 - q1
    threshold = q3 + factor * iqr
    return threshold

def save_threshold(threshold, path="threshold_value.npy"):
    np.save(path, threshold)

def load_threshold(path="threshold_value.npy"):
    return np.load(path)
