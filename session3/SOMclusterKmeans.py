#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import mode
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class SOMKohonen:
    """
    A simple Self-Organizing Map (Kohonen Map) for clustering.
    This class supports a 2D grid of neurons, each with a weight vector in 'input_dim'.
    """

    def __init__(
        self, 
        map_size=(10, 10),   # shape of the SOM grid
        input_dim=2,         # dimension of input data
        learning_rate=0.5,
        sigma=3.0,           # initial neighborhood radius
        lr_decay=0.99,       # learning rate decay per epoch
        sigma_decay=0.99,    # sigma decay per epoch
        num_final_clusters=3,
        seed=None
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
        self.num_final_clusters = num_final_clusters  

        # Initialize SOM weights: shape = [rows, cols, input_dim]
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)
        self.cluster_labels = None  
        self.bmu_cluster_map = {}  # Mapping BMUs to clusters

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
            self.learning_rate *= self.lr_decay
            self.sigma *= self.sigma_decay
        
        # Apply clustering to consolidate BMUs into final groups
        self._cluster_bmus(data)

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
                dist_sq = (i - bmu_row)**2 + (j - bmu_col)**2
                neigh_strength = np.exp(-dist_sq / (2.0 * (self.sigma**2)))
                self.weights[i, j, :] += self.learning_rate * neigh_strength * (x - self.weights[i, j, :])

    def get_cluster_assignments(self, data):
        """
        Find the BMU for each data point and return a list of (bmu_row, bmu_col).
        """
        return [self._find_bmu(x) for x in data]

    def _cluster_bmus(self, data):
        """
        Clusters the BMUs into a predefined number of final clusters using K-Means.
        """
        bmu_positions = np.array(list(set(self.get_cluster_assignments(data))))  
        
        kmeans = KMeans(n_clusters=self.num_final_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(bmu_positions)

        self.bmu_cluster_map = {tuple(bmu): cluster for bmu, cluster in zip(bmu_positions, self.cluster_labels)}

    def _assign_closest_existing_bmu(self, bmu):
        """
        If a BMU is not found in self.bmu_cluster_map, find the closest known BMU.
        """
        bmu_array = np.array(list(self.bmu_cluster_map.keys()))
        distances = cdist([bmu], bmu_array, metric='euclidean')
        closest_index = np.argmin(distances)
        return tuple(bmu_array[closest_index])

    def plot_clusters(self, train_data, test_data=None):
        """
        Plot training data clusters with optional test data.
        """
        train_data = np.array(train_data)
        train_assignments = self.get_cluster_assignments(train_data)

        # Assign colors to training clusters
        color_labels = [self.bmu_cluster_map[bmu] for bmu in train_assignments]

        plt.figure()
        scatter = plt.scatter(train_data[:,0], train_data[:,1], c=color_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Final Cluster ID")

        rows, cols, _ = self.weights.shape
        w_2d = self.weights.reshape(rows*cols, 2)
        plt.scatter(w_2d[:,0], w_2d[:,1], marker='s', s=30, c='black', label="SOM Neurons")

        if test_data is not None:
            test_data = np.array(test_data)
            test_assignments = self.get_cluster_assignments(test_data)

            test_colors = [self.bmu_cluster_map.get(bmu, self.bmu_cluster_map[self._assign_closest_existing_bmu(bmu)]) 
                           for bmu in test_assignments]

            plt.scatter(test_data[:,0], test_data[:,1], marker='*', s=100, c=test_colors, cmap='tab10', edgecolors='black', label="Test Data")

        plt.legend()
        plt.title("SOM Clustering with Final Cluster Consolidation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def evaluate_test_data(self, test_data, true_labels):
        """
        Evaluates clustering performance by comparing predicted labels with true labels.
        """
        predicted_assignments = self.get_cluster_assignments(test_data)
        predicted_labels = np.array([self.bmu_cluster_map.get(bmu, self.bmu_cluster_map[self._assign_closest_existing_bmu(bmu)])
                                     for bmu in predicted_assignments])

        cluster_purity = 0
        for i in range(self.num_final_clusters):
            cluster_indices = np.where(predicted_labels == i)[0]
            if len(cluster_indices) > 0:
                most_common_value = mode(true_labels[cluster_indices], keepdims=False)[0]
                most_common = most_common_value[0] if isinstance(most_common_value, np.ndarray) else most_common_value
                cluster_purity += np.sum(true_labels[cluster_indices] == most_common)

        purity_score = cluster_purity / len(true_labels)
        print(f"Clustering Purity Score: {purity_score:.2f}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== SOM (Kohonen Map) for 2D Clustering with Test Data ===")

    np.random.seed(42)
    train_data = np.vstack([
        np.random.normal(loc=(0,0), scale=0.5, size=(100,2)),
        np.random.normal(loc=(5,5), scale=0.8, size=(120,2)),
        np.random.normal(loc=(2,8), scale=0.7, size=(80,2))
    ])
    train_labels = np.concatenate([[0]*100, [1]*120, [2]*80])

    test_data = np.vstack([
        np.random.normal(loc=(0,0), scale=0.5, size=(10,2)),
        np.random.normal(loc=(5,5), scale=0.8, size=(10,2)),
        np.random.normal(loc=(2,8), scale=0.7, size=(10,2))
    ])
    test_labels = np.concatenate([[0]*10, [1]*10, [2]*10])

    som = SOMKohonen(map_size=(10,10), input_dim=2, seed=123)
    som.train(train_data, epochs=30)
    som.plot_clusters(train_data, test_data)
    som.evaluate_test_data(test_data, test_labels)
