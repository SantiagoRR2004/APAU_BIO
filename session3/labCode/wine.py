import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


import PSO
import ACO


def plot_test_data_clusters(X_Train, y_Train, X_Test, y_test):

    plt.figure(figsize=(8, 6))

    # Plot training data
    plt.scatter(
        X_Train[:, 0],
        X_Train[:, 1],
        c=X_Test,
        cmap="viridis",
        marker="o",
        alpha=0.3,
        label="Training Data",
    )

    # Plot test data with same color as assigned cluster
    plt.scatter(
        X_Train[:, 0],
        X_Train[:, 1],
        c=y_test,
        cmap="viridis",
        marker="s",
        edgecolors="black",
        s=80,
        label="Test Data",
    )

    plt.title("Clustering Results (Test Data)")
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
