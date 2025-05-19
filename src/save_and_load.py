import os
import numpy as np

def save_processed_data(X_train, y_train, X_test, y_test, file_prefix="processed_data"):
    """
    Saves processed data (numpy arrays) to the processed data folder.

    Parameters:
        X_train, y_train: Training data and labels (processed).
        X_test, y_test: Testing data and labels (processed).
        file_prefix (str): Filename prefix for the processed data files.
    """
    processed_path = "../data/processed/"
    os.makedirs(processed_path, exist_ok=True)
    
    np.save(os.path.join(processed_path, f"{file_prefix}_X_train.npy"), X_train)
    np.save(os.path.join(processed_path, f"{file_prefix}_y_train.npy"), y_train)
    np.save(os.path.join(processed_path, f"{file_prefix}_X_test.npy"), X_test)
    np.save(os.path.join(processed_path, f"{file_prefix}_y_test.npy"), y_test)

    print(f"Processed data saved to {processed_path}")

def load_processed_data(file_prefix="processed_data"):
    """
    Loads processed data from the processed data folder.

    Parameters:
        file_prefix (str): Filename prefix for the processed data files.

    Returns:
        tuple: X_train, y_train, X_test, y_test (numpy arrays).
    """
    processed_path = "../data/processed/"
    
    X_train = np.load(os.path.join(processed_path, f"{file_prefix}_X_train.npy"))
    y_train = np.load(os.path.join(processed_path, f"{file_prefix}_y_train.npy"))
    X_test = np.load(os.path.join(processed_path, f"{file_prefix}_X_test.npy"))
    y_test = np.load(os.path.join(processed_path, f"{file_prefix}_y_test.npy"))
    
    print(f"Loaded processed data from {processed_path}")
    return X_train, y_train, X_test, y_test
