import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.stats import mode


import PSO
import ACO


def plot_clusters(train_data, test_data=None, test_labels=None, cluster_labels=None):
    """
    Plots clustered training data and optionally test data.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        train_data[:, 0],
        train_data[:, 1],
        c=cluster_labels,
        cmap="tab10",
        edgecolors="k",
        marker="o",
        label="Train Data",
    )

    if test_data is not None and test_labels is not None:
        plt.scatter(
            test_data[:, 0],
            test_data[:, 1],
            c=test_labels,
            cmap="tab10",
            edgecolors="black",
            marker="*",
            s=100,
            label="Test Data",
        )

    plt.title("ACO-Based Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def evaluate_test_data(true_labels, predicted_labels):
    """Evaluates clustering performance by comparing predicted labels with true labels."""

    # Compute clustering purity using majority vote
    cluster_purity = 0
    if isinstance(true_labels, pd.core.series.Series):
        true_labels = true_labels.to_numpy()

    for i in range(len(np.unique(predicted_labels))):
        cluster_indices = np.where(predicted_labels == i)[0]
        if len(cluster_indices) > 0:
            most_common_value = mode(true_labels[cluster_indices], keepdims=False)[0]
            if isinstance(
                most_common_value, np.ndarray
            ):  # Handle cases where mode() returns an array
                most_common = most_common_value[0]
            else:
                most_common = most_common_value  # If it's a scalar, use it directly
            cluster_purity += np.sum(true_labels[cluster_indices] == most_common)

    purity_score = cluster_purity / len(true_labels)  # Fix NameError
    print(f"Clustering Purity Score: {purity_score:.2f}")


if __name__ == "__main__":
    seed = 0
    # Cargamos el dataset de vinos
    wine = load_wine()
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    wine_df["cultivar"] = wine.target

    # Información general
    print(wine_df.head())
    print(wine_df.shape)
    print(wine_df["cultivar"].value_counts())

    # Preparamos los datos
    X = wine_df.drop("cultivar", axis=1)
    y = wine_df["cultivar"]
    tsne = TSNE(n_components=2, random_state=seed)
    data_2D = tsne.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        data_2D, y, test_size=0.1, random_state=seed
    )

    dust = PSO.PSOfunction(X_train, X_test, seed)
    ants = ACO.ACOfunction(X_train, X_test, seed)

    print("PSO", sep=" ")
    evaluate_test_data(y_test, dust)

    print("ACO", sep=" ")
    evaluate_test_data(y_test, ants)
