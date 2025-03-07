import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


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
