import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_images(X):
    """
    Preprocesses the image data by normalizing pixel values.

    Parameters:
        X (ndarray): The image data (flattened).

    Returns:
        ndarray: The preprocessed image data (normalized).
    """
    print("\n🔄 Normalizing image data...")
    X = X / 255.0
    print("Image data normalized.")
    return X

def preprocess_labels(y):
    """
    Converts numeric labels to one-hot encoded format.

    Parameters:
        y (ndarray): The label data (numeric).

    Returns:
        ndarray: The one-hot encoded labels.
    """
    print("\nConverting labels to one-hot encoding...")
    
    # Get the unique number of labels (e.g., 26 for EMNIST letters)
    num_classes = len(np.unique(y))
    
    # One-hot encoding using NumPy (faster, simpler)
    y_encoded = np.eye(num_classes)[y - 1]  # Subtract 1 to make labels 0-based

    print(f"Labels one-hot encoded. Shape: {y_encoded.shape}")
    return y_encoded
