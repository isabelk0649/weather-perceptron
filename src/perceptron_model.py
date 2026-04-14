import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the path of the current script to find the data folder
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "../data/weather_2025.csv")

def load_data():
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure the CSV is in the /data folder.")
        return None
    return pd.read_csv(csv_path)

def predict(X, weights, bias):
    return np.where(np.dot(X, weights) + bias >= 0, 1, 0)

def main():
    data = load_data()
    if data is None: return

    # Features and Target
    X = data[['temp', 'humid']].values
    y = data['rain'].values

    # Train-Test Split
    X_train, y_train = X[:15], y[:15]
    X_test, y_test = X[15:], y[15:]

    # Model Initialization
    np.random.seed(42) # For reproducibility
    weights = np.random.uniform(-1, 1, size=X.shape[1])
    bias = np.random.uniform(-1, 1)
    learning_rate = 0.1
    epochs = 100

    # Training Loop
    for epoch in range(epochs):
        for i in range(len(X_train)):
            prediction = predict(X_train[i], weights, bias)
            error = y_train[i] - prediction
            weights += learning_rate * error * X_train[i]
            bias += learning_rate * error

    # Plotting Results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='s', label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o', edgecolors='black', label='Test')
    
    # Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.title('Perceptron Decision Boundary (Weather Prediction)')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Normalized Humidity')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
