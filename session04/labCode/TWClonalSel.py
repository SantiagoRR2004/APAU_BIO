from manage_data import get_data
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
# In case of failure:
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from TWClonalSelV4 import ClonalSelectionAIS
from timeseries import plot_timeseries_with_windows


def from_data_to_timeseries(
    data,
    anomaly_intervals=[],
    window_size=100,
    step=20,
):
    T = data["value"]
    # Slice into windows
    window_starts = range(0, len(T) - window_size + 1, step)
    X, y = [], []
    for ws in window_starts:
        we = ws + window_size
        window_data = T[ws:we]
        # label=1 if overlaps any anomaly interval
        label = 0
        for a_start, a_end in anomaly_intervals:
            if not (we <= a_start or ws >= a_end):
                label = 1
                break
        X.append(window_data)
        y.append(label)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    return T, X, y, list(window_starts)


def plot_result(ais, X, y, pca):
    plt.figure(figsize=(8, 6))
    plt.title("Final AIS Centers vs. Data")
    X2 = pca.transform(X)
    y_pred = ais.predict(X)
    plt.scatter(
        X2[y_pred == 0, 0],
        X2[y_pred == 0, 1],
        c="blue",
        alpha=0.5,
        label="Pred: Normal",
    )
    plt.scatter(
        X2[y_pred == 1, 0],
        X2[y_pred == 1, 1],
        c="red",
        alpha=0.5,
        label="Pred: Anomaly",
    )
    anomalies_idx = np.where(y == 1)[0]
    plt.scatter(
        X2[anomalies_idx, 0],
        X2[anomalies_idx, 1],
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=80,
        label="True Anomaly",
    )
    plt.legend()
    plt.show()


def loss_history(ais):
    plt.figure()
    plt.plot(ais.coverage_loss_history_, marker="o")
    plt.title("AIS - Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Sum of Min Distances (Loss)")
    plt.grid()
    plt.show()


def main():
    normalData, anomalyData = get_data()
    step = 20
    window_size = 100

    T_normal, X_normal, y_normal, window_starts_normal = from_data_to_timeseries(
        normalData, anomaly_intervals=[], window_size=window_size, step=step
    )
    plot_timeseries_with_windows(
        T_normal,
        anomaly_intervals=[],
        window_size=window_size,
        window_starts=window_starts_normal,
        y=y_normal,
    )

    # Get the anormal interval
    anomaly_idx = np.argwhere(anomalyData["value"] >= 90)[:, 0]
    anomaly_interval = [anomaly_idx[0], anomaly_idx[-1]]
    print("Anomaly Interval:", anomaly_interval)

    T, X, y, window_starts = from_data_to_timeseries(
        anomalyData,
        anomaly_intervals=[anomaly_interval],
        window_size=window_size,
        step=step,
    )
    plot_timeseries_with_windows(
        T,
        anomaly_intervals=[anomaly_interval],
        window_size=window_size,
        window_starts=window_starts,
        y=y,
    )

    pca = PCA(n_components=2, random_state=777)
    pca.fit(X)

    ais = ClonalSelectionAIS(
        pop_size=30,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.2,
        max_gens=50,
        diversity_rate=0.1,
        random_seed=123,
    )

    ais.fit(X_normal, pca)

    loss_history(ais)

    y_pred = ais.predict(X)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))

    plot_result(ais, X, y, pca)

    # Testeamos con otro
    _, anomalyData = get_data(anomaly="art_daily_jumpsdown.csv")

    T, X, y, window_starts = from_data_to_timeseries(
        anomalyData,
        anomaly_intervals=[anomaly_interval],
        window_size=window_size,
        step=step,
    )
    plot_timeseries_with_windows(
        T,
        anomaly_intervals=[anomaly_interval],
        window_size=window_size,
        window_starts=window_starts,
        y=y,
    )

    y_pred = ais.predict(X)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))

    plot_result(ais, X, y, pca)


if __name__ == "__main__":
    main()
