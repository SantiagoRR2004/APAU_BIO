#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from scipy.stats import mode


# Define the Particle class
class Particle:
    def __init__(self, num_clusters, data):
        self.num_clusters = num_clusters
        self.position = np.random.rand(
            num_clusters, data.shape[1]
        )  # Initialize random cluster centroids
        self.velocity = np.random.rand(
            num_clusters, data.shape[1]
        )  # Initialize velocity
        self.best_position = np.copy(self.position)
        self.best_error = np.inf
        self.error = np.inf

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        inertia = w * self.velocity
        cognitive = (
            c1
            * np.random.rand(self.num_clusters, self.position.shape[1])
            * (self.best_position - self.position)
        )
        social = (
            c2
            * np.random.rand(self.num_clusters, self.position.shape[1])
            * (global_best_position - self.position)
        )
        self.velocity = inertia + cognitive + social

    def update_position(self):
        self.position += self.velocity

    def evaluate(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.position, axis=2)
        labels = np.argmin(distances, axis=1)
        # Calculate clustering error (sum of squared distances)
        self.error = np.sum(np.min(distances, axis=1) ** 2)
        if self.error < self.best_error:
            self.best_error = self.error
            self.best_position = np.copy(self.position)


# PSO for clustering
class PSO:
    def __init__(self, num_clusters, data, num_particles=30, max_iters=100, seed=None):
        self.num_clusters = num_clusters
        self.data = data
        self.num_particles = num_particles
        self.max_iters = max_iters
        self.global_best_position = None
        self.global_best_error = np.inf
        self.particles = [Particle(num_clusters, data) for _ in range(num_particles)]

        if seed is not None:
            np.random.seed(seed)

    def optimize(self):
        for i in range(self.max_iters):
            for particle in self.particles:
                particle.evaluate(self.data)
                if particle.best_error < self.global_best_error:
                    self.global_best_error = particle.best_error
                    self.global_best_position = np.copy(particle.best_position)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            print(
                f"Iteration {i+1}/{self.max_iters}, Best Error: {self.global_best_error}"
            )

        return self.global_best_position

    def assign_clusters(self, data):
        """Assigns each data point to the nearest cluster centroid."""
        distances = np.linalg.norm(
            data[:, np.newaxis] - self.global_best_position, axis=2
        )
        labels = np.argmin(distances, axis=1)
        return labels

    def plot_clusters(self):
        """Plots clustering results for the training data."""
        labels = self.assign_clusters(self.data)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.data[:, 0],
            self.data[:, 1],
            c=labels,
            cmap="viridis",
            marker="o",
            label="Training Data",
        )
        plt.scatter(
            self.global_best_position[:, 0],
            self.global_best_position[:, 1],
            c="red",
            marker="x",
            s=200,
            label="Centroids",
        )
        plt.title("PSO Clustering Results (Training Data)")
        plt.legend()
        plt.show()

    def plot_test_data_clusters(self, test_data, true_labels):
        """Plots test data and its assigned clusters using the trained PSO model."""
        train_labels = self.assign_clusters(self.data)
        test_labels = self.assign_clusters(test_data)

        plt.figure(figsize=(8, 6))

        # Plot training data
        plt.scatter(
            self.data[:, 0],
            self.data[:, 1],
            c=train_labels,
            cmap="viridis",
            marker="o",
            alpha=0.3,
            label="Training Data",
        )

        # Plot test data with same color as assigned cluster
        plt.scatter(
            test_data[:, 0],
            test_data[:, 1],
            c=test_labels,
            cmap="viridis",
            marker="s",
            edgecolors="black",
            s=80,
            label="Test Data",
        )

        # Plot centroids
        plt.scatter(
            self.global_best_position[:, 0],
            self.global_best_position[:, 1],
            c="red",
            marker="x",
            s=200,
            label="Centroids",
        )

        plt.title("PSO Clustering Results (Test Data)")
        plt.legend()
        plt.show()

    def evaluate_test_data(self, test_data, true_labels):
        """Evaluates clustering performance by comparing predicted labels with true labels."""
        predicted_labels = self.assign_clusters(test_data)

        # Compute clustering purity using majority vote
        cluster_purity = 0
        if isinstance(true_labels, pd.core.series.Series):
            true_labels = true_labels.to_numpy()

        for i in range(self.num_clusters):
            cluster_indices = np.where(predicted_labels == i)[0]
            if len(cluster_indices) > 0:
                most_common_value = mode(true_labels[cluster_indices], keepdims=False)[
                    0
                ]
                if isinstance(
                    most_common_value, np.ndarray
                ):  # Handle cases where mode() returns an array
                    most_common = most_common_value[0]
                else:
                    most_common = most_common_value  # If it's a scalar, use it directly
                cluster_purity += np.sum(true_labels[cluster_indices] == most_common)

        purity_score = cluster_purity / len(true_labels)  # Fix NameError
        print(f"Clustering Purity Score: {purity_score:.2f}")


# Generate synthetic training data
def generate_data(num_samples=200, centers=3):
    X, y, centers = make_blobs(
        n_samples=num_samples,
        centers=centers,
        cluster_std=1.0,
        random_state=42,
        return_centers=True,
    )
    return X, y, centers


# Generate test data sampled from the **same original clusters**
def generate_test_data(num_samples=50, centers=None):
    X, y = make_blobs(
        n_samples=num_samples, centers=centers, cluster_std=1.0, random_state=99
    )
    return X, y


# Main function
def main():
    # Generate training and test data
    data, labels, centers = generate_data(num_samples=200, centers=3)
    test_data, test_labels = generate_test_data(num_samples=50, centers=centers)

    # Initialize and run PSO for clustering
    pso = PSO(num_clusters=3, data=data, num_particles=30, max_iters=100)
    pso.optimize()

    # Plot training and test clustering results
    pso.plot_clusters()
    pso.plot_test_data_clusters(test_data, test_labels)

    # Evaluate test clustering performance
    pso.evaluate_test_data(test_data, test_labels)


if __name__ == "__main__":
    main()
