# src/feature_engineering.py

import numpy as np
from scipy.ndimage import rotate, shift, zoom, sobel
from skimage.transform import resize

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
    Extracts basic and advanced pixel-based features for each image.

    Parameters:
        X (ndarray): The image data (flattened).

    Returns:
        ndarray: Array of extracted features.
    """
    print("\nExtracting pixel intensity features...")
    X_reshaped = X.reshape(-1, 28, 28)
    
    # Basic statistics
    mean_values = np.mean(X_reshaped, axis=(1, 2))
    std_values = np.std(X_reshaped, axis=(1, 2))
    max_values = np.max(X_reshaped, axis=(1, 2))
    min_values = np.min(X_reshaped, axis=(1, 2))

    # Regional statistics (split image into 4 quadrants)
    Q1 = X_reshaped[:, :14, :14]
    Q2 = X_reshaped[:, :14, 14:]
    Q3 = X_reshaped[:, 14:, :14]
    Q4 = X_reshaped[:, 14:, 14:]

    region_means = np.mean(Q1, axis=(1, 2)), np.mean(Q2, axis=(1, 2)), \
                   np.mean(Q3, axis=(1, 2)), np.mean(Q4, axis=(1, 2))
    
    # Edge detection (Sobel filter)
    sobel_x = np.array([sobel(image, axis=0) for image in X_reshaped])
    sobel_y = np.array([sobel(image, axis=1) for image in X_reshaped])
    edge_intensity = np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2), axis=(1, 2))
    
    # Binary pixel count (thresholding)
    binary_count = np.sum(X_reshaped > 0.5, axis=(1, 2))  # Assuming normalized (0-1)
    
    # Symmetry analysis (Horizontal and Vertical)
    horizontal_symmetry = np.mean(X_reshaped == np.flip(X_reshaped, axis=2), axis=(1, 2))
    vertical_symmetry = np.mean(X_reshaped == np.flip(X_reshaped, axis=1), axis=(1, 2))
    
    # Combining features
    features = np.hstack([
        mean_values.reshape(-1, 1),
        std_values.reshape(-1, 1),
        max_values.reshape(-1, 1),
        min_values.reshape(-1, 1),
        np.vstack(region_means).T,
        edge_intensity.reshape(-1, 1),
        binary_count.reshape(-1, 1),
        horizontal_symmetry.reshape(-1, 1),
        vertical_symmetry.reshape(-1, 1)
    ])
    
    print(f"Extracted pixel features. Shape: {features.shape}")
    return features

def augment_images(X, y, augment_factor=2):
    """
    Augments image data by rotating, shifting, and zooming images.

    Parameters:
        X (ndarray): The image data (reshaped to 28x28 or flattened).
        y (ndarray): The corresponding labels.
        augment_factor (int): The number of augmentations to create for each image.

    Returns:
        tuple: Augmented image data and labels.
    """
    print("\nAugmenting images...")
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        image = X[i]
        label = y[i]

        # Ensure the image is 2D (28x28)
        if image.ndim == 1:
            image = image.reshape(28, 28)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image.reshape(28, 28)  # Remove channel dimension if present

        # Add the original image
        X_augmented.append(image)
        y_augmented.append(label)

        # Generate augmentations
        for _ in range(augment_factor):
            # Random rotation
            rotated = rotate(image, angle=np.random.uniform(-15, 15), reshape=False)
            rotated = resize(rotated, (28, 28), anti_aliasing=True)

            # Random shifting
            shifted = shift(image, shift=(np.random.uniform(-2, 2), np.random.uniform(-2, 2)))
            shifted = resize(shifted, (28, 28), anti_aliasing=True)

            # Random zooming
            zoomed = zoom(image, zoom=(np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)))
            zoomed = resize(zoomed, (28, 28), anti_aliasing=True)

            # Append augmented images (normalized)
            X_augmented.append(np.clip(rotated, 0, 1))
            y_augmented.append(label)

            X_augmented.append(np.clip(shifted, 0, 1))
            y_augmented.append(label)

            X_augmented.append(np.clip(zoomed, 0, 1))
            y_augmented.append(label)

    print(f"Augmented {len(X)} images to {len(X_augmented)} images.")
    return np.array(X_augmented), np.array(y_augmented)
