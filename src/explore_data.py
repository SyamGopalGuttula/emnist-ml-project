# src/explore_data.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def explore_class_distribution(y, title="Class Distribution"):
    """
    Visualizes the distribution of classes in the dataset.

    Parameters:
        y (ndarray): Label array.
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(x=y, hue=y, dodge=False, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.savefig("../report/class_distribution.png")
    plt.show()

def analyze_pixel_distribution(X):
    """
    Analyzes the distribution of pixel values in the dataset.

    Parameters:
        X (ndarray): Image data (flattened).
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(X.flatten(), bins=50, kde=True, color="blue")
    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig("../report/pixel_value_distribution.png")
    plt.show()

def visualize_sample_images(X, y, samples=10):
    """
    Visualizes a set of sample images from the dataset.

    Parameters:
        X (ndarray): Image data (reshaped).
        y (ndarray): Corresponding labels.
        samples (int): Number of samples to display.
    """
    plt.figure(figsize=(15, 5))
    for i in range(samples):
        plt.subplot(1, samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.savefig("../report/sample_images.png")
    plt.show()

def correlation_heatmap(X, sample_size=500):
    """
    Displays a correlation heatmap of pixel values (optimized).

    Parameters:
        X (ndarray): Image data (flattened).
        sample_size (int): Number of images to sample for heatmap.
    """
    print("\nGenerating Correlation Heatmap...")

    # Randomly sample images for speed
    X_sample = X[:sample_size].reshape(sample_size, -1)

    # Remove columns with zero variance
    zero_variance_columns = np.std(X_sample, axis=0) == 0
    X_sample = X_sample[:, ~zero_variance_columns]

    if X_sample.shape[1] == 0:
        print("All columns have zero variance. Correlation Heatmap cannot be generated.")
        return

    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(X_sample, rowvar=False)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='viridis', cbar=True)
    plt.title("Correlation Heatmap (Pixel Intensity)")
    plt.savefig("../report/correlation_heatmap.png")
    plt.show()


def missing_value_analysis(X):
    """
    Checks for missing values in the dataset.

    Parameters:
        X (ndarray): Image data (flattened).
    """
    missing_values = np.sum(np.isnan(X))
    print(f"Missing Values in the Dataset: {missing_values}")
    if missing_values > 0:
        print("Warning: There are missing values in the data.")
    else:
        print("No missing values detected.")
