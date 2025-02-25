#!/usr/bin/env python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode

class AntColonyClustering:
    """
    Ant Colony Optimization for Clustering.
    Ants reinforce paths between similar points, forming clusters.
    """

    def __init__(
        self,
        distance_matrix,
        num_ants=10,
        alpha=1.0,             # Importance of pheromone
        beta=2.0,              # Importance of similarity
        evaporation_rate=0.5,  # Pheromone evaporation
        pheromone_constant=100,# Q in ACO
        generations=50,
        num_clusters=3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.distance_matrix = np.array(distance_matrix)
        self.num_nodes = self.distance_matrix.shape[0]
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = pheromone_constant
        self.generations = generations
        self.num_clusters = num_clusters  

        # Initialize pheromone trails
        self.pheromones = np.ones((self.num_nodes, self.num_nodes)) * 0.1

        # Heuristic (1/distance), except diagonal = 0
        self.desirability = 1.0 / (self.distance_matrix + 1e-9)
        np.fill_diagonal(self.desirability, 0.0)

        self.cluster_labels = None  # Will store final cluster assignments

    def run(self):
        """
        Main ACO loop for clustering.
        """
        for gen in range(self.generations):
            all_paths = []
            for _ in range(self.num_ants):
                path = self._construct_path()
                all_paths.append(path)

            # Update pheromones based on ant paths
            self._update_pheromones(all_paths)

            print(f"Generation {gen+1}/{self.generations} - Pheromone Update Complete")

        # Cluster formation: Convert pheromones into final clusters
        self.cluster_labels = self._form_clusters()
        return self.cluster_labels

    def _construct_path(self):
        """
        Each ant builds a tour by selecting next nodes probabilistically.
        """
        unvisited = list(range(self.num_nodes))
        start = random.choice(unvisited)
        path = [start]
        unvisited.remove(start)

        current_node = start
        while unvisited:
            next_node = self._select_next_node(current_node, unvisited)
            path.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        return path

    def _select_next_node(self, current_node, unvisited):
        """
        Selects the next node based on pheromone strength and heuristic information.
        """
        pheromone_vals = np.array([self.pheromones[current_node, j] ** self.alpha for j in unvisited])
        heuristic_vals = np.array([self.desirability[current_node, j] ** self.beta for j in unvisited])

        probabilities = pheromone_vals * heuristic_vals
        probabilities /= probabilities.sum()

        return np.random.choice(unvisited, p=probabilities)

    def _update_pheromones(self, all_paths):
        """
        Evaporates old pheromones and deposits new ones based on paths.
        """
        self.pheromones *= (1 - self.evaporation_rate)

        for path in all_paths:
            deposit_amount = self.Q / len(path)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromones[a, b] += deposit_amount
                self.pheromones[b, a] += deposit_amount

    def _form_clusters(self):
        """
        Convert pheromone matrix into clusters using Agglomerative Clustering.
        """
        # Convert pheromone matrix into similarity (inverse of distance)
        similarity_matrix = self.pheromones / self.pheromones.max()

        clustering = AgglomerativeClustering(n_clusters=self.num_clusters, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(1 - similarity_matrix)  # Convert similarity to distance

        return labels

    def assign_test_data(self, test_data, train_data):
        """
        Assigns test data to clusters based on nearest neighbor in training data.
        """
        test_distances = cdist(test_data, train_data, metric='euclidean')
        nearest_train_indices = np.argmin(test_distances, axis=1)
        test_labels = self.cluster_labels[nearest_train_indices]
        return test_labels

    def evaluate_test_data(self, test_labels, true_test_labels):
        """
        Evaluates clustering performance by comparing predicted labels with true labels.
        """
        cluster_purity = 0
        unique_clusters = np.unique(test_labels)

        for cluster in unique_clusters:
            cluster_indices = np.where(test_labels == cluster)[0]
            if len(cluster_indices) > 0:
                most_common_value = mode(true_test_labels[cluster_indices], keepdims=False)[0]
                most_common = most_common_value[0] if isinstance(most_common_value, np.ndarray) else most_common_value
                cluster_purity += np.sum(true_test_labels[cluster_indices] == most_common)

        purity_score = cluster_purity / len(true_test_labels)
        print(f"Clustering Purity Score: {purity_score:.2f}")

    def plot_clusters(self, train_data, test_data=None, test_labels=None):
        """
        Plots clustered training data and optionally test data.
        """
        plt.figure(figsize=(8,6))
        plt.scatter(train_data[:, 0], train_data[:, 1], c=self.cluster_labels, cmap='tab10', edgecolors='k', marker='o', label="Train Data")

        if test_data is not None and test_labels is not None:
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='tab10', edgecolors='black', marker='*', s=100, label="Test Data")

        plt.title("ACO-Based Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Ant Colony Optimization for Clustering with Test Validation ===")

    # Generate synthetic training data (3 clusters)
    np.random.seed(42)
    train_data = np.vstack([
        np.random.normal(loc=(0,0), scale=0.5, size=(100,2)),
        np.random.normal(loc=(5,5), scale=0.8, size=(120,2)),
        np.random.normal(loc=(2,8), scale=0.7, size=(80,2))
    ])
    train_labels = np.concatenate([[0]*100, [1]*120, [2]*80])

    # Generate test data sampled from the same distributions
    test_data = np.vstack([
        np.random.normal(loc=(0,0), scale=0.5, size=(10,2)),
        np.random.normal(loc=(5,5), scale=0.8, size=(10,2)),
        np.random.normal(loc=(2,8), scale=0.7, size=(10,2))
    ])
    test_labels = np.concatenate([[0]*10, [1]*10, [2]*10])

    # Compute distance matrix for training data
    train_distance_matrix = cdist(train_data, train_data, metric='euclidean')

    # Instantiate ACO clustering
    aco_clustering = AntColonyClustering(
        distance_matrix=train_distance_matrix,
        num_ants=10,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.3,
        pheromone_constant=100,
        generations=20,
        num_clusters=3,
        seed=42
    )

    # Run ACO clustering
    cluster_labels = aco_clustering.run()

    # Assign test data to clusters
    test_labels_predicted = aco_clustering.assign_test_data(test_data, train_data)

    # Evaluate test clustering performance
    aco_clustering.evaluate_test_data(test_labels_predicted, test_labels)

    # Plot results
    aco_clustering.plot_clusters(train_data, test_data, test_labels_predicted)
