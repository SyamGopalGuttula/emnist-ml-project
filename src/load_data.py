import pandas as pd
import os

def load_emnist_data(subset="balanced"):
    """
    Loads the EMNIST dataset based on the chosen subset.

    Parameters:
        subset (str): The EMNIST subset to load ("balanced", "byclass", "letters", etc.)

    Returns:
        tuple: X_train, y_train, X_test, y_test (numpy arrays)
    """
    data_path = "../data/raw/"

    if subset == "balanced":
        train_path = os.path.join(data_path, "emnist-balanced-train.csv")
        test_path = os.path.join(data_path, "emnist-balanced-test.csv")
    elif subset == "byclass":
        train_path = os.path.join(data_path, "emnist-byclass-train.csv")
        test_path = os.path.join(data_path, "emnist-byclass-test.csv")
    elif subset == "letters":
        train_path = os.path.join(data_path, "emnist-letters-train.csv")
        test_path = os.path.join(data_path, "emnist-letters-test.csv")
    else:
        raise ValueError("Invalid subset. Choose from 'balanced', 'byclass', 'letters'.")

    # Load data using pandas
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    # Separate features and labels
    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    print(f"Loaded EMNIST {subset} subset.")
    print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test
