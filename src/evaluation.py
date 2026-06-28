import numpy as np
import pandas as pd

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_nmae(y_true, y_pred):
    """Calculate Normalized Mean Absolute Error (NMAE)."""
    mae = calculate_mae(y_true, y_pred)
    y_mean = np.mean(y_true)
    if y_mean == 0:
        return 0
    return mae / y_mean

def calculate_mase(y_true, y_pred, y_naive):
    """Calculate Mean Absolute Scaled Error (MASE)."""
    mae_model = calculate_mae(y_true, y_pred)
    mae_naive = calculate_mae(y_true, y_naive)
    if mae_naive == 0:
        return 0
    return mae_model / mae_naive

def calculate_accuracy(nmae):
    """Calculate Operational Accuracy."""
    acc = (1 - nmae) * 100
    # Floor at 0%
    return max(0.0, acc)
