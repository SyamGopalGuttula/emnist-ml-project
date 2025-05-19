import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def explore_class_distribution(y, title="Class Distribution"):
    """
    Visualizes the class distribution of the labels.

    Parameters:
        y (ndarray): The label data (one-hot encoded or numeric).
        title (str): Title of the plot.
    """
    print("\n Analyzing Class Distribution...")
    
    # If one-hot encoded, convert to numeric
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1) + 1  # Convert one-hot to numeric

    plt.figure(figsize=(12, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class (Letter)")
    plt.ylabel("Frequency")
    plt.show()

def analyze_pixel_distribution(X):
    """
    Analyzes the pixel value distribution of the images.

    Parameters:
        X (ndarray): The image data (flattened, normalized).
    """
    print("\n Analyzing Pixel Value Distribution...")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(X.flatten(), bins=50, color='blue')
    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Intensity (0 to 1)")
    plt.ylabel("Frequency")
    plt.show()

def visualize_sample_images(X, y, samples=5):
    """
    Visualizes a few sample images with labels.

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
        plt.title(f"Label: {np.argmax(y[i]) + 1}")  # Convert one-hot to label
        plt.axis('off')
    plt.show()

def correlation_heatmap(X, sample_size=1000):
    """
    Displays a correlation heatmap for a sample of pixel values.

    Parameters:
        X (ndarray): Image data (flattened).
        sample_size (int): Number of samples to use for the heatmap.
    """
    print("\n Generating Correlation Heatmap...")
    
    # Take a small sample to speed up calculation
    X_sample = X[:sample_size]
    X_df = pd.DataFrame(X_sample)
    
    plt.figure(figsize=(12, 10))
    corr_matrix = X_df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Pixel Value Correlation Heatmap")
    plt.show()

def missing_value_analysis(X):
    """
    Analyzes missing values in the dataset.

    Parameters:
        X (ndarray): Image data (flattened).
    """
    print("\n Checking for Missing Values...")
    missing_values = np.isnan(X).sum()
    print(f"Total Missing Values: {missing_values}")
