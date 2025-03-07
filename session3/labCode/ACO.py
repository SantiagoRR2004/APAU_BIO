import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the AntColonyClustering module
import AntColonyClustering


import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


def ACOfunction(X_train, X_test, seed) -> list:

    distance_matrix = cdist(X_train, X_train, metric="euclidean")

    # Initialize and run ACO for clustering
    ant = AntColonyClustering.AntColonyClustering(
        distance_matrix=distance_matrix, seed=seed
    )

    # Run ACO clustering
    ant.run()

    # Assign test data to clusters
    return ant.assign_test_data(X_test, X_train)


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

    distance_matrix = cdist(X_train, X_train, metric="euclidean")

    # Initialize and run PSO for clustering
    ant = AntColonyClustering.AntColonyClustering(
        distance_matrix=distance_matrix,
    )

    # Run ACO clustering
    cluster_labels = ant.run()

    # Assign test data to clusters
    test_labels_predicted = ant.assign_test_data(X_test, X_train)

    # Plot results
    ant.plot_clusters(X_train, X_test, test_labels_predicted)
