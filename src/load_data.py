# src/load_data.py

import pandas as pd
import numpy as np

def load_emnist_data(train_path: str, test_path: str):
    """
    Loads the EMNIST dataset from CSV files.

    Parameters:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        tuple: X_train, y_train, X_test, y_test (NumPy arrays)
    """
    print("Loading EMNIST dataset...")
    
    # Load CSV files using pandas
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Separate features and labels
    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    print(f"Data loaded. Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test
