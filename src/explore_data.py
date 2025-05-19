import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def explore_data(X_train, y_train, X_test, y_test):
    """
    Displays basic information about the training and testing data.

    Parameters:
        X_train (ndarray): Training images.
        y_train (ndarray): Training labels.
        X_test (ndarray): Testing images.
        y_test (ndarray): Testing labels.
    """
    print("Exploring Data...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    
    print("\n Displaying first few labels (Training):")
    print(y_train[:10])

    print("\n Unique labels:", np.unique(y_train))
    print(" Unique labels in test set:", np.unique(y_test))

def visualize_sample_images(X, y, samples=5):
    """
    Visualizes a few sample images from the dataset.

    Parameters:
        X (ndarray): Image data (flattened).
        y (ndarray): Labels corresponding to the images.
        samples (int): Number of samples to display.
    """
    print(f"\n Visualizing {samples} sample images...")
    plt.figure(figsize=(10, 5))
    for i in range(samples):
        plt.subplot(1, samples, i + 1)
        image = X[i].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()
