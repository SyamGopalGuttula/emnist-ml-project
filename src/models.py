import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, model_name):
    """
    Saves a trained model to the models/ directory.

    Parameters:
        model: Trained model object.
        model_name (str): Name of the model file to save.
    """
    model_path = os.path.join('../models', model_name)
    joblib.dump(model, model_path)
    print(f"Model saved as: {model_path}")

def load_model(model_name):
    """
    Loads a trained model from the models/ directory.

    Parameters:
        model_name (str): Name of the model file to load.

    Returns:
        model: Loaded model object.
    """
    model_path = os.path.join('../models', model_name)
    model = joblib.load(model_path)
    print(f"Loaded model: {model_name}")
    return model

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the given training data.

    Parameters:
        X_train (ndarray): Training feature data.
        y_train (ndarray): Training labels.

    Returns:
        model: Trained Logistic Regression model.
    """
    print("\nTraining Logistic Regression...")

    # Scale the data (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize Logistic Regression with optimized settings
    model = LogisticRegression(
        solver='saga',        # More robust for high-dimensional data
        max_iter=1000,        # Increased iterations for convergence
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    save_model(model, "logistic_regression.pkl")
    print(f"Model saved as: ../models/logistic_regression.pkl")

    return model

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors (KNN) model on the training data.

    Parameters:
        X_train (ndarray): Training feature data (flattened or features).
        y_train (ndarray): Training labels (one-hot encoded or numeric).
        n_neighbors (int): Number of neighbors for KNN.

    Returns:
        model: Trained KNN model.
    """
    print("\nTraining K-Nearest Neighbors (KNN)...")
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    save_model(model, "knn.pkl")
    return model

def train_decision_tree(X_train, y_train, max_depth=None):
    """
    Trains a Decision Tree model on the training data.

    Parameters:
        X_train (ndarray): Training feature data (flattened or features).
        y_train (ndarray): Training labels (one-hot encoded or numeric).
        max_depth (int): Maximum depth of the tree (None for unlimited).

    Returns:
        model: Trained Decision Tree model.
    """
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    save_model(model, "decision_tree.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on the test data.

    Parameters:
        model: Trained model (Logistic, KNN, or Decision Tree).
        X_test (ndarray): Testing feature data.
        y_test (ndarray): Testing labels.

    Returns:
        None
    """
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)

    # If the labels are one-hot encoded, convert to numeric
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
