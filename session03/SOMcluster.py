#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import mode


class SOMKohonen:
    """
    A simple Self-Organizing Map (Kohonen Map) for clustering.
    This class supports a 2D grid of neurons, each with a weight vector in 'input_dim'.
    """

    def __init__(
        self,
        map_size=(10, 10),  # shape of the SOM grid
        input_dim=2,  # dimension of input data
        learning_rate=0.5,
        sigma=3.0,  # initial neighborhood radius
        lr_decay=0.99,  # learning rate decay per epoch
        sigma_decay=0.99,  # sigma decay per epoch
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.map_size = map_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.lr_decay = lr_decay
        self.sigma_decay = sigma_decay

        # Initialize SOM weights: shape = [rows, cols, input_dim]
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def train(self, data, epochs=100):
        """
        Train the SOM using the given data for the specified number of epochs.
        """
        data = np.array(data)
        for epoch in range(epochs):
            np.random.shuffle(data)
            for x in data:
                bmu_row, bmu_col = self._find_bmu(x)
                self._update_weights(x, bmu_row, bmu_col)
            # Decay learning rate, sigma
            self.learning_rate *= self.lr_decay
            self.sigma *= self.sigma_decay

    def _find_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) for input x.
        Returns (row, col) of the BMU in the grid.
        """
        diff = self.weights - x
        dist_sq = np.sum(diff**2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        return bmu_idx

    def _update_weights(self, x, bmu_row, bmu_col):
        """
        Update weights of BMU and its neighbors using Gaussian neighborhood function.
        """
        rows, cols, _ = self.weights.shape
        for i in range(rows):
            for j in range(cols):
                dist_sq = (i - bmu_row) ** 2 + (j - bmu_col) ** 2
                neigh_strength = np.exp(-dist_sq / (2.0 * (self.sigma**2)))
                self.weights[i, j, :] += (
                    self.learning_rate * neigh_strength * (x - self.weights[i, j, :])
                )

    def get_cluster_assignments(self, data):
        """
        Find the BMU for each data point and return a list of (bmu_row, bmu_col).
        """
        assignments = []
        for x in data:
            bmu_row, bmu_col = self._find_bmu(x)
            assignments.append((bmu_row, bmu_col))
        return assignments

    def plot_clusters(self, train_data, test_data=None, true_labels=None):
        """
        Plot training data clusters with optional test data.
        """
        if self.input_dim != 2:
            print("plot_clusters() is only implemented for 2D data.")
            return

        train_data = np.array(train_data)
        train_assignments = self.get_cluster_assignments(train_data)

        # Assign unique colors to each BMU cluster
        unique_ids = {}
        next_id = 0
        color_labels = []
        for key in train_assignments:
            if key not in unique_ids:
                unique_ids[key] = next_id
                next_id += 1
            color_labels.append(unique_ids[key])

        plt.figure()
        scatter = plt.scatter(
            train_data[:, 0], train_data[:, 1], c=color_labels, cmap="tab10", alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster ID (BMU)")

        # SOM Neurons
        rows, cols, _ = self.weights.shape
        w_2d = self.weights.reshape(rows * cols, 2)
        plt.scatter(
            w_2d[:, 0], w_2d[:, 1], marker="s", s=30, c="black", label="SOM Neurons"
        )

        # Plot test data if provided
        if test_data is not None:
            test_data = np.array(test_data)
            test_assignments = self.get_cluster_assignments(test_data)

            # Assign test points the same color as their BMU cluster
            test_colors = [unique_ids[key] for key in test_assignments]

            plt.scatter(
                test_data[:, 0],
                test_data[:, 1],
                marker="*",
                s=100,
                c=test_colors,
                cmap="tab10",
                edgecolors="black",
                label="Test Data",
            )

        plt.legend()
        plt.title("SOM Clustering with Test Data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def evaluate_test_data(self, test_data, true_labels):
        """
        Evaluates clustering performance by comparing predicted labels with true labels.
        """
        predicted_assignments = self.get_cluster_assignments(test_data)

        # Compute clustering purity using majority vote
        cluster_purity = 0
        unique_clusters = list(set(predicted_assignments))
        cluster_mapping = {
            c: i for i, c in enumerate(unique_clusters)
        }  # Assign cluster indices

        predicted_labels = np.array([cluster_mapping[c] for c in predicted_assignments])

        for i in range(len(unique_clusters)):
            cluster_indices = np.where(predicted_labels == i)[0]
            if len(cluster_indices) > 0:
                most_common_value = mode(true_labels[cluster_indices], keepdims=False)[
                    0
                ]
                most_common = (
                    most_common_value[0]
                    if isinstance(most_common_value, np.ndarray)
                    else most_common_value
                )
                cluster_purity += np.sum(true_labels[cluster_indices] == most_common)

        purity_score = cluster_purity / len(true_labels)
        print(f"Clustering Purity Score: {purity_score:.2f}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== SOM (Kohonen Map) for 2D Clustering with Test Data ===")

    # Generate synthetic training data (3 clusters)
    np.random.seed(42)
    blob1 = np.random.normal(loc=(0, 0), scale=0.5, size=(100, 2))
    blob2 = np.random.normal(loc=(5, 5), scale=0.8, size=(120, 2))
    blob3 = np.random.normal(loc=(2, 8), scale=0.7, size=(80, 2))
    train_data = np.vstack([blob1, blob2, blob3])
    train_labels = np.concatenate(
        [[0] * 100, [1] * 120, [2] * 80]
    )  # Ground truth labels

    # Generate test data (sampled from same clusters)
    test_blob1 = np.random.normal(loc=(0.2, 0.1), scale=0.5, size=(10, 2))
    test_blob2 = np.random.normal(loc=(5.2, 5.2), scale=0.8, size=(10, 2))
    test_blob3 = np.random.normal(loc=(2.1, 7.8), scale=0.7, size=(10, 2))
    test_data = np.vstack([test_blob1, test_blob2, test_blob3])
    test_labels = np.concatenate(
        [[0] * 10, [1] * 10, [2] * 10]
    )  # True labels for test data

    # Create and train SOM
    som = SOMKohonen(
        map_size=(10, 10),
        input_dim=2,
        learning_rate=0.5,
        sigma=3.0,
        lr_decay=0.95,
        sigma_decay=0.95,
        seed=123,
    )
    som.train(train_data, epochs=30)

    # Plot clustering result with test data
    som.plot_clusters(train_data, test_data, true_labels=test_labels)

    # Evaluate test clustering performance
    som.evaluate_test_data(test_data, test_labels)
