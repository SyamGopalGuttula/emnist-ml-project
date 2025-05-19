# src/feature_engineering.py

import numpy as np
from scipy.ndimage import rotate, shift, zoom

def reshape_images(X):
    """
    Reshapes flattened image data to 28x28 (for model input).

    Parameters:
        X (ndarray): The image data (flattened).

    Returns:
        ndarray: Reshaped image data (28x28).
    """
    print("\nReshaping images to 28x28...")
    return X.reshape(-1, 28, 28, 1)  # Adding channel dimension for CNN

def extract_pixel_features(X):
    """
    Extracts statistical pixel-based features (mean, std) for each image.

    Parameters:
        X (ndarray): The image data (flattened).

    Returns:
        ndarray: Array of extracted features.
    """
    print("\nExtracting pixel intensity features...")
    X_reshaped = X.reshape(-1, 28, 28)
    
    # Calculating pixel statistics
    mean_values = np.mean(X_reshaped, axis=(1, 2))
    std_values = np.std(X_reshaped, axis=(1, 2))
    max_values = np.max(X_reshaped, axis=(1, 2))
    min_values = np.min(X_reshaped, axis=(1, 2))

    # Combining features
    features = np.stack([mean_values, std_values, max_values, min_values], axis=1)
    print(f"Extracted pixel features. Shape: {features.shape}")
    return features

def augment_images(X, y, augment_factor=2):
    """
    Augments image data by rotating and shifting images.

    Parameters:
        X (ndarray): The image data (reshaped to 28x28).
        y (ndarray): The corresponding labels.
        augment_factor (int): The number of augmentations to create for each image.

    Returns:
        tuple: Augmented image data and labels.
    """
    print("\n Augmenting images...")
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        image = X[i]
        label = y[i]

        # Add the original image
        X_augmented.append(image)
        y_augmented.append(label)

        # Generate augmentations
        for _ in range(augment_factor):
            rotated = rotate(image, angle=np.random.uniform(-15, 15), reshape=False)
            shifted = shift(image, shift=(np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0))
            X_augmented.append(rotated)
            X_augmented.append(shifted)
            y_augmented.append(label)
            y_augmented.append(label)

    print(f" Augmented {len(X)} images to {len(X_augmented)} images.")
    return np.array(X_augmented), np.array(y_augmented)