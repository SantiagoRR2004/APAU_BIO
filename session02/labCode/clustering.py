import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from sklearn.datasets import load_wine
import pandas as pd
import random
import numpy as np
from lab02code import GeneticAlgorithm


class MlTaskClustering:
    """
    Esta clase debe contener los datos y definir cómo:
      - crear un individuo
      - calcular el fitness
      - hacer crossover
      - mutar los individuos
    """

    def __init__(self, data, k=3, seed=None):
        """
        Inicializa la clase con los datos y la semilla
        k: número de clusters
        """
        if seed:
            random.seed(seed)
        self.data = data
        self.k = k
        self.n_features = data.shape[1]
        self.vector_size = k * self.n_features
        self.lower_bounds = [min(data[:, i]) for i in range(self.n_features)]
        self.upper_bounds = [max(data[:, i]) for i in range(self.n_features)]

    def create_individual(self):
        """
        Crea un individuo aleatorio
        """
        individual = []
        for _ in range(self.k):
            cluster_center = []
            for i in range(self.n_features):
                cluster_center.append(
                    random.uniform(self.lower_bounds[i], self.upper_bounds[i])
                )
            individual.extend(cluster_center)
        return tuple(individual)

    def fitness_function(self, individual):
        """
        Calcula el fitness de un individuo
        """
        # Reshape centers_vals into (k, n_features) array
        centers = np.array(individual).reshape(self.k, self.n_features)

        # Sum of squared distances
        # for each point, find nearest center
        points = self.data
        # distances shape => (#points, #centers)
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        # find min distance for each point
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)

        return -sse  # negate to maximize

    def crossover(self, p1, p2, crossover_rate=0.8):
        """
        Cruza dos individuos
        """
        if random.random() > crossover_rate:
            return p1, p2
        alpha = random.random()
        child1 = tuple(alpha * x1 + (1 - alpha) * x2 for x1, x2 in zip(p1, p2))
        child2 = tuple(alpha * x2 + (1 - alpha) * x1 for x1, x2 in zip(p1, p2))
        return child1, child2

    def mutation(self, individual, mutation_rate=0.1):
        """
        Muta el cluster de un individuo
        """
        if random.random() > mutation_rate:
            return individual
        else:
            cluster_index = random.randint(0, self.k - 1)
            new_cluster = []
            for i in range(self.n_features):
                new_cluster.append(
                    random.uniform(self.lower_bounds[i], self.upper_bounds[i])
                )
            new_individual = list(individual)
            new_individual[
                cluster_index * self.n_features : cluster_index * self.n_features
                + self.n_features
            ] = new_cluster
            return tuple(new_individual)

    def plot_clusters(self, centers, y=None):
        """
        Visualiza los clusters
        """
        if y is not None:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=y)
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100)
        plt.show()

    def generate_table(self):
        """
        Genera una tabla con los mejores individuos y sus fitnesses.
        """
        try:
            from prettytable import PrettyTable

            table = PrettyTable()
            features = []
            for k in range(self.k):
                for i in range(self.n_features):
                    features.append(f"x{k}_{i}")
            table.field_names = ["Gen", "Fitness"] + features
        except ImportError:
            table = None
        return table

    def update_table(self, table, gen, best_ind, best_fit):
        """
        Actualiza la tabla con los mejores individuos y sus fitnesses.
        """
        features = []
        for k in range(self.k):
            for i in range(self.n_features):
                features.append(f"x{k}_{i}")
        features_values = [round(x, 3) for x in best_ind]
        table.add_row([gen, round(best_fit, 3)] + features_values)


if __name__ == "__main__":
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
    tsne = TSNE(n_components=2, random_state=42)
    data_2D = tsne.fit_transform(X)

    # Corremos el algoritmo genético
    seed = 42
    k = 3
    ml_task = MlTaskClustering(data_2D, k=k, seed=seed)
    ga = GeneticAlgorithm(seed=seed)
    best_individual = ga.run(ml_task=ml_task)
    centers = np.array(best_individual).reshape(k, 2)
    ml_task.plot_clusters(centers, y)
