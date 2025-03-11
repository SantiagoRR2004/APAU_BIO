from manage_data import get_data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
# In case of failure:
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from AISclonalSelectionV4 import ClonalAISAnomaly
from timeseries import plot_timeseries_with_windows


def loss_history(ais):
    plt.figure()
    plt.plot(ais.loss_history_, marker="o")
    plt.title("AIS - Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Sum of Min Distances (Loss)")
    plt.grid()
    plt.show()


def main():
    normalData, anomalyData = get_data()

    # Create a train set
    X_train = normalData.copy()
    X_train["timestamp"] = np.arange(len(X_train))
    X_train = X_train.to_numpy(dtype=np.float64)
    print(X_train[:5])

    # Create a test set
    X_test = anomalyData.copy()
    X_test["timestamp"] = np.arange(len(X_test))
    X_test = X_test.to_numpy(dtype=np.float64)
    print(X_test[:5])

    # Get the anormal interval
    anomaly_idx = np.argwhere(anomalyData["value"] >= 90)[:, 0]
    anomaly_interval = [anomaly_idx[0], anomaly_idx[-1]]
    print("Anomaly Interval:", anomaly_interval)

    y_test = np.zeros(len(anomalyData))
    y_test[anomaly_idx] = 1

    ais = ClonalAISAnomaly(
        pop_size=50,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.1,
        n_generations=25,
        diversity_rate=0.15,
        random_seed=42,
        visualize=True,
    )

    ais.fit(X_train)

    loss_history(ais)

    # ----------------------------------------------------
    # 4. Prediction & Evaluation
    # ----------------------------------------------------
    y_pred = ais.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))


if __name__ == "__main__":
    main()
