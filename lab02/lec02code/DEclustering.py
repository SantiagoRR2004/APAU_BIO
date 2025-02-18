#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
import Clustering


class ClusteringDE(Clustering.Clustering):
    """
    Differential Evolution (DE) for K-means-like clustering:
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
        F=0.5,  # Mutation factor
        CR=0.9,  # Crossover probability
        patience=10,  # Early stopping patience
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
        :param F: Differential weight (mutation factor)
        :param CR: Crossover probability in [0,1]
        :param patience: Early stopping if no SSE improvement for 'patience' consecutive generations
        :param min_delta: Minimum SSE improvement threshold to reset patience
        :param seed: Optional random seed
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.k = k
        self.dim = dim
        self.params_per_ind = k * dim

        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
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

        # Initialize population
        self.population = self._create_initial_population()
        # Evaluate SSE -> fitness = -SSE for all
        self.fitness_vals = [self._fitness_function(ind) for ind in self.population]

    # ------------------------------------------------------------
    # (2) Population Initialization
    # ------------------------------------------------------------
    def _create_initial_population(self):
        """
        Create initial population of size pop_size.
        Each individual is a NumPy array (k*dim) of center coordinates.
        """
        population = []
        for _ in range(self.pop_size):
            ind = np.random.uniform(
                low=self.lower_bound, high=self.upper_bound, size=self.params_per_ind
            )
            population.append(ind)
        return population

    # ------------------------------------------------------------
    # (3) Fitness Function = -SSE
    # ------------------------------------------------------------
    def _fitness_function(self, individual):
        """
        Negative SSE: for each point, find nearest center among the K centers in 'individual'.
        """
        centers = individual.reshape(self.k, self.dim)
        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)
        return -sse

    def _calculate_sse(self, individual):
        """
        Just the SSE (for logging, final comparison, etc.).
        """
        centers = individual.reshape(self.k, self.dim)
        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)
        return sse

    # ------------------------------------------------------------
    # (4) Differential Evolution Steps
    # ------------------------------------------------------------
    def _mutation(self, idx):
        """
        DE/rand/1: For the target idx, pick r1, r2, r3 distinct from idx.
        v = x[r1] + F * (x[r2] - x[r3])
        """
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        r1, r2, r3 = random.sample(candidates, 3)

        x_r1 = self.population[r1]
        x_r2 = self.population[r2]
        x_r3 = self.population[r3]

        # Mutation
        v = x_r1 + self.F * (x_r2 - x_r3)

        # Ensure donor is within bounds (optional; some DE variants let them go out of bounds)
        v = np.clip(v, self.lower_bound, self.upper_bound)
        return v

    def _crossover(self, target_vec, donor_vec):
        """
        Binomial crossover:
        trial[j] = donor[j] if rand < CR or j == rand_index
        else trial[j] = target_vec[j]
        """
        trial = np.copy(target_vec)
        rand_index = random.randint(0, self.params_per_ind - 1)
        for j in range(self.params_per_ind):
            if random.random() < self.CR or j == rand_index:
                trial[j] = donor_vec[j]
        return trial

    # ------------------------------------------------------------
    # (5) Main DE Loop
    # ------------------------------------------------------------
    def run(self):
        best_sse = float("inf")
        no_improve_count = 0

        # track best global
        global_best_ind = None
        global_best_fit = -float("inf")
        global_best_sse = float("inf")

        for gen in range(self.max_generations):
            for i in range(self.pop_size):
                # 1) Mutation
                donor = self._mutation(i)
                # 2) Crossover
                trial = self._crossover(self.population[i], donor)

                # 3) Selection
                trial_fitness = self._fitness_function(trial)
                if trial_fitness > self.fitness_vals[i]:
                    # if trial is better, replace
                    self.population[i] = trial
                    self.fitness_vals[i] = trial_fitness

            # Evaluate best in this generation
            gen_best_idx = np.argmax(self.fitness_vals)
            gen_best_fit = self.fitness_vals[gen_best_idx]
            gen_best_ind = self.population[gen_best_idx]
            gen_best_sse = self._calculate_sse(gen_best_ind)

            # Update global best
            if gen_best_sse < global_best_sse:
                global_best_ind = gen_best_ind
                global_best_fit = gen_best_fit
                global_best_sse = gen_best_sse

            # Early stopping
            if gen_best_sse < best_sse - self.min_delta:
                best_sse = gen_best_sse
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print(f"Early stopping at generation {gen} (no SSE improvement).")
                break

            # Log progress
            print(f"Gen {gen}: Best SSE = {gen_best_sse:.4f}")

        # Final check among population
        final_best_idx = np.argmax(self.fitness_vals)
        final_best_fit = self.fitness_vals[final_best_idx]
        final_best_ind = self.population[final_best_idx]
        final_best_sse = self._calculate_sse(final_best_ind)

        if global_best_ind is not None and global_best_sse < final_best_sse:
            truly_best = global_best_ind
            truly_best_fit = global_best_fit
            truly_best_sse = global_best_sse
        else:
            truly_best = final_best_ind
            truly_best_fit = final_best_fit
            truly_best_sse = final_best_sse

        print("\n=== DE Clustering: Final Best Individual ===")
        print(f"Best Fitness (=-SSE): {truly_best_fit:.4f}")
        print(f"SSE: {truly_best_sse:.4f}")
        print(f"Cluster centers:\n{truly_best.reshape(self.k, self.dim)}")

        self._plot_clusters(truly_best)
        return truly_best

    # ------------------------------------------------------------
    # (6) Plotting (only for 2D)
    # ------------------------------------------------------------
    def _plot_clusters(self, best_individual):
        """
        If dim=2, visualize final clustering (points colored by nearest center).
        """
        if self.dim != 2:
            print("Plotting is only implemented for 2D.")
            return

        centers = best_individual.reshape(self.k, 2)
        points = self.data

        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        fig, ax = plt.subplots()
        for c_idx in range(self.k):
            cluster_pts = points[cluster_assignments == c_idx]
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=30, alpha=0.6)

        ax.scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X", s=150, edgecolor="k"
        )
        ax.set_title("Differential Evolution Clustering (2D)")
        plt.show()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Differential Evolution for Clustering Demo ===")

    # Example usage: clustering 3 groups in 2D
    de_clustering = ClusteringDE(
        data=None,  # or provide your own (N x 2) data
        k=3,
        dim=2,
        pop_size=50,
        lower_bound=-10,
        upper_bound=10,
        max_generations=50,
        F=0.5,
        CR=0.9,
        patience=8,
        min_delta=1e-2,
        seed=42,
    )

    best_solution = de_clustering.run()
