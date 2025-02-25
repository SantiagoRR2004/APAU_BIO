#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.stats import mode

class BeeColonyClustering:
    """
    Improved Bee Colony Optimization (BCO) for Clustering with:
      - Bounding cluster centers to data range
      - Final K-Means on top solutions to reduce fragmentation
      - Refined local search to converge smoothly
    """

    def __init__(
        self,
        data,
        num_clusters=3,
        num_bees=30,
        num_employed=15,
        num_scouts=5,
        generations=50,
        top_solutions=5,  # number of best solutions to keep for final K-Means
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.data = np.array(data)
        self.num_samples, self.num_features = self.data.shape
        self.num_clusters = num_clusters

        self.num_bees = num_bees
        self.num_employed = num_employed
        self.num_scouts = num_scouts
        self.generations = generations
        self.top_solutions = top_solutions

        # Determine bounding box of data to prevent centroids from drifting away
        self.data_min = self.data.min(axis=0)
        self.data_max = self.data.max(axis=0)

        # Initialize random positions for bee solutions (shape: [num_bees, num_clusters, num_features])
        self.positions = np.random.uniform(
            low=self.data_min, high=self.data_max,
            size=(self.num_bees, self.num_clusters, self.num_features)
        )

        # Track best solution
        self.best_solution = None
        self.best_fitness = np.inf

    def fitness(self, centroids):
        """
        Compute sum of squared distances of all data points to their nearest centroid.
        """
        distances = cdist(self.data, centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def _bound_solution(self, solution):
        """
        Clip centroid positions to stay within the data bounding box.
        """
        return np.clip(solution, self.data_min, self.data_max)

    def _local_search(self, current_position):
        """
        Generate new position near current_position, guided by the global best solution.
        """
        # Slightly move toward the global best
        direction = (self.best_solution - current_position)
        step = np.random.uniform(-0.1, 0.1, current_position.shape) * direction
        new_position = current_position + step
        return self._bound_solution(new_position)

    def update_positions(self):
        """
        Core BCO steps:
         1) Employed bees refine their solutions.
         2) Onlooker bees choose top solutions and refine them.
         3) Scout bees explore random positions near best solution.
        """

        # === 1) Employed Bees refine (local search)
        for i in range(self.num_employed):
            new_position = self._local_search(self.positions[i])
            new_fitness = self.fitness(new_position)
            if new_fitness < self.fitness(self.positions[i]):
                self.positions[i] = new_position

        # === 2) Onlooker Bees pick solutions based on fitness
        fitness_values = np.array([self.fitness(pos) for pos in self.positions])
        # Probability is inversely related to fitness
        probabilities = np.exp(-fitness_values)
        prob_sum = probabilities.sum()
        if prob_sum == 0 or np.isnan(prob_sum):
            probabilities = np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities /= prob_sum

        selected_indices = np.random.choice(
            range(self.num_bees), size=self.num_employed, p=probabilities
        )
        for i, idx in enumerate(selected_indices):
            candidate = self._local_search(self.positions[idx])
            if self.fitness(candidate) < self.fitness(self.positions[self.num_employed + i]):
                self.positions[self.num_employed + i] = candidate

        # === 3) Scout Bees explore random positions near best solution
        for i in range(self.num_scouts):
            # Move the last i-th bee
            random_explore = self.best_solution + np.random.uniform(-1.0, 1.0, self.best_solution.shape)
            self.positions[-(i+1)] = self._bound_solution(random_explore)

    def _finalize_centroids(self, top_positions):
        """
        Consolidate final centroids by performing K-Means on the best solutions.
        """
        # Flatten top solutions to shape (top_solutions * num_clusters, num_features)
        all_centroids = top_positions.reshape(-1, self.num_features)

        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
        kmeans.fit(all_centroids)
        return kmeans.cluster_centers_

    def run(self):
        """
        Main Bee Colony Optimization loop.
        """
        # Initialize best solution from the lowest fitness among positions
        initial_fitnesses = [self.fitness(pos) for pos in self.positions]
        best_idx = np.argmin(initial_fitnesses)
        self.best_fitness = initial_fitnesses[best_idx]
        self.best_solution = self.positions[best_idx].copy()

        # === Main Loop
        for gen in range(self.generations):
            self.update_positions()

            # Evaluate best in the population
            fitnesses = np.array([self.fitness(pos) for pos in self.positions])
            current_best_idx = np.argmin(fitnesses)
            current_best_fit = fitnesses[current_best_idx]
            if current_best_fit < self.best_fitness:
                self.best_fitness = current_best_fit
                self.best_solution = self.positions[current_best_idx].copy()

            print(f"Gen {gen+1}/{self.generations} | Best Fitness: {self.best_fitness:.2f}")

        # === Final Consolidation
        # 1) Sort all positions by fitness
        fitnesses = np.array([self.fitness(pos) for pos in self.positions])
        sorted_indices = np.argsort(fitnesses)
        top_indices = sorted_indices[:self.top_solutions]
        top_positions = self.positions[top_indices]

        # 2) K-Means on the top solutions
        final_centroids = self._finalize_centroids(top_positions)
        self.best_solution = final_centroids
        self.best_fitness = self.fitness(final_centroids)

        print(f"\nBCO Finished | Final Best Fitness: {self.best_fitness:.2f}")
        return self.best_solution

    def assign_test_data(self, test_data):
        """
        Assign test data to clusters based on nearest centroid in self.best_solution.
        """
        test_distances = cdist(test_data, self.best_solution, metric='euclidean')
        test_labels = np.argmin(test_distances, axis=1)
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
        train_labels = np.argmin(cdist(train_data, self.best_solution, metric='euclidean'), axis=1)

        plt.figure(figsize=(8,6))
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='tab10', edgecolors='k', marker='o', label="Train Data")

        if test_data is not None and test_labels is not None:
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='tab10', edgecolors='black', marker='*', s=100, label="Test Data")

        # Plot final centroids
        plt.scatter(self.best_solution[:, 0], self.best_solution[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title("Improved BCO Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Improved Bee Colony Optimization for Clustering with Test Validation ===")

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

    # Instantiate BCO clustering
    bco_clustering = BeeColonyClustering(
        data=train_data,
        num_clusters=3,
        num_bees=30,
        num_employed=15,
        num_scouts=5,
        generations=20,
        top_solutions=5,   # We'll keep 5 best solutions for final K-Means
        seed=42
    )

    # Run BCO clustering
    cluster_centroids = bco_clustering.run()

    # Assign test data to clusters
    test_labels_predicted = bco_clustering.assign_test_data(test_data)

    # Evaluate test clustering performance
    bco_clustering.evaluate_test_data(test_labels_predicted, test_labels)

    # Plot results
    bco_clustering.plot_clusters(train_data, test_data, test_labels_predicted)
