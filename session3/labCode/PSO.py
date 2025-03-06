import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the PSOcluster module
from PSOcluster import PSO

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


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

    # Initialize and run PSO for clustering
    pso = PSO(num_clusters=3, data=X_train, num_particles=50, max_iters=150, seed=seed)
    pso.optimize()

    # Plot training and test clustering results
    pso.plot_clusters()
    pso.plot_test_data_clusters(X_test, y_test)

    # Evaluate test clustering performance
    pso.evaluate_test_data(X_test, y_test)
