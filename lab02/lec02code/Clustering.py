#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

# Import abstract
from abc import ABC, abstractmethod


class Clustering(ABC):
    """
    Abstract class for K-means-like clustering:
    Minimizes SSE from points to cluster centers.
    """

    def __init__(
        self,
        data=None,  # If None, generate random 2D blobs
        k=3,  # Number of clusters
        dim=2,  # Data dimensionality
        pop_size=50,
        lower_bound=-10,
        upper_bound=10,
        max_generations=50,
        patience=100,  # Early stopping patience
        min_delta=1e-3,  # Minimum improvement in SSE
        seed=None,
    ):
        """
        :param data: (N x dim) array of data points; if None, we generate random 2D blobs for demonstration.
        :param k: Number of clusters
        :param dim: Dimensionality of the data
        :param pop_size: Number of individuals in the population
        :param lower_bound: Minimum coordinate for each cluster center
        :param upper_bound: Maximum coordinate for each cluster center
        :param max_generations: Max number of DE iterations
        :param patience: Early stopping if no SSE improvement for 'patience' consecutive generations
        :param min_delta: Minimum SSE improvement threshold to reset patience
        :param seed: Optional random seed
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # For K cluster centers, each center has 'dim' coordinates => total = k * dim
        self.k = k
        self.dim = dim
        self.vector_size = k * dim  # length of the cluster-center vector

        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_generations = max_generations
        self.patience = patience
        self.min_delta = min_delta

        # Data handling
        if data is None:
            self.data = self._generate_gaussian_blobs(
                num_points_per_blob=60, centers=[(0, 0), (5, 5), (0, 5)], std=1.0
            )
        else:
            self.data = np.array(data)
            if self.data.shape[1] != dim:
                raise ValueError(
                    f"Data dimension ({self.data.shape[1]}) does not match 'dim' ({dim})."
                )

        self.bestSSEByGeneration = []

    # ------------------------------------------------------------
    # (1) Generate Synthetic Data (Optional)
    # ------------------------------------------------------------
    def _generate_gaussian_blobs(
        self, num_points_per_blob=60, centers=[(0, 0), (5, 5)], std=1.0
    ):
        """
        Generate synthetic 2D data from multiple Gaussian blobs.
        """
        all_points = []
        for cx, cy in centers:
            blob = np.random.normal(
                loc=(cx, cy), scale=std, size=(num_points_per_blob, 2)
            )
            all_points.append(blob)
        data = np.vstack(all_points)
        return data

    # ------------------------------------------------------------
    # (2) Create Individual
    # ------------------------------------------------------------

    @abstractmethod
    def create_individual(self):
        pass

    # ------------------------------------------------------------
    # (4) Population Initialization
    # ------------------------------------------------------------

    def create_initial_population(self) -> list:
        population = []
        for _ in range(self.pop_size):
            population.append(self.create_individual())
        return population


if __name__ == "__main__":
    from DEclustering import ClusteringDE
    from GPclustering import ClusteringGP
    from GAclustering import ClusteringGA

    data = ClusteringDE()._generate_gaussian_blobs()
    seed = 42

    DE = ClusteringDE(data=data, seed=seed)
    DE.run()

    GP = ClusteringGP(data=data, seed=seed)
    GP.run()

    GA = ClusteringGA(data=data, seed=seed)
    GA.run()

    print(max(DE.bestSSEByGeneration))
    print(max(GP.bestSSEByGeneration))
    print(max(GA.bestSSEByGeneration))

    # Plotting in one single figure, no subplots

    plt.title("Clustering SSE by Generation")
    plt.xlabel("Generation")
    plt.ylabel("SSE")

    plt.plot(DE.bestSSEByGeneration, label="Differential Evolution")
    plt.plot(GP.bestSSEByGeneration, label="Genetic Programming")
    plt.plot(GA.bestSSEByGeneration, label="Genetic Algorithm")

    plt.legend()
    plt.show()
