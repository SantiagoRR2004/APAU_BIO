#!/usr/bin/env python

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import Clustering


class ClusteringGA(Clustering.Clustering):
    """
    Genetic Algorithm for clustering data into K clusters.
    Can use either real-coded or binary-coded representation
    for the cluster center coordinates.
    """

    def __init__(
        self,
        use_binary_representation=False,
        chrom_length=10,  # bits per coordinate if binary-coded
        mutation_rate=0.05,
        elitism=True,
        *args,
        **kwargs,
    ):
        """
        :param use_binary_representation: If True, use binary-coded representation; otherwise real-coded.
        :param chrom_length: length of each coordinate in bits (used only if binary-coded).
        :param mutation_rate: base mutation probability
        :param elitism: keep the best individual from current gen into next gen
        """
        super().__init__(*args, **kwargs)

        self.use_binary = use_binary_representation
        self.chrom_length = chrom_length
        self.mutation_rate = mutation_rate
        self.elitism = elitism

    # ------------------------------------------------------------
    # (1) GA Representation
    # ------------------------------------------------------------

    def encode_real_to_binary(self, real_values):
        """
        Convert the real-coded array of length (k*dim) into a bitstring (list of 0/1).
        Each coordinate is encoded into self.chrom_length bits.
        """
        bitstring = []
        for val in real_values:
            bits = self._float_to_bits(val)
            bitstring.extend(bits)
        return bitstring

    def decode_binary_to_real(self, bitstring):
        """
        Convert a bitstring into a list (or tuple) of k*dim real coordinates.
        Each chunk is self.chrom_length bits => total length = self.params_per_ind * self.chrom_length.
        """
        chunk_size = self.chrom_length
        real_values = []
        for i in range(self.params_per_ind):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = bitstring[start:end]
            val = self._bits_to_float(chunk)
            real_values.append(val)
        return tuple(real_values)

    def _float_to_bits(self, val):
        """
        Map float val in [lower_bound, upper_bound] -> a chunk of length self.chrom_length bits.
        """
        val = max(min(val, self.upper_bound), self.lower_bound)
        max_bits_val = (1 << self.chrom_length) - 1  # 2^self.chrom_length - 1

        # Normalize in [0,1]
        norm = (val - self.lower_bound) / (self.upper_bound - self.lower_bound)

        # Map to integer in [0, max_bits_val]
        int_val = int(round(norm * max_bits_val))

        # Convert int_val to bit list (LSB->MSB) then reverse
        bits = [(int_val >> i) & 1 for i in range(self.chrom_length)]
        bits.reverse()
        return bits

    def _bits_to_float(self, bits):
        """
        Convert chunk of bits -> float in [lower_bound, upper_bound].
        bits[0] is MSB, bits[-1] is LSB if reversed above.
        """
        val_int = 0
        for b in bits:
            val_int = (val_int << 1) | b

        max_bits_val = (1 << self.chrom_length) - 1
        norm = val_int / max_bits_val
        real_val = self.lower_bound + norm * (self.upper_bound - self.lower_bound)
        return real_val

    # ------------------------------------------------------------
    # (2) Create Individual
    # ------------------------------------------------------------

    def create_individual(self):
        """
        Create a single individual. If real-coded, it's a tuple of length k*dim.
        If binary-coded, it's a bitstring of length (k*dim * chrom_length).
        """
        if self.use_binary:
            # internally create random real-coded first, then encode
            real_vals = [
                random.uniform(self.lower_bound, self.upper_bound)
                for _ in range(self.params_per_ind)
            ]
            return self.encode_real_to_binary(real_vals)
        else:
            # real-coded
            return tuple(
                random.uniform(self.lower_bound, self.upper_bound)
                for _ in range(self.vector_size)
            )

    # ------------------------------------------------------------
    # (3) Fitness Function (Clustering SSE)
    # ------------------------------------------------------------

    def fitness_function(self, individual):
        """
        For the given individual (which encodes K cluster centers),
        compute the sum of squared distances (SSE) from each data point to its nearest cluster center.
        We *negate* SSE so that a lower SSE => higher (less negative) fitness.
        """
        # decode if binary-coded
        if self.use_binary:
            centers_vals = self.decode_binary_to_real(individual)
        else:
            centers_vals = individual

        # Reshape centers_vals into (k, dim)
        centers = np.array(centers_vals).reshape(self.k, self.dim)

        # Sum of squared distances
        # for each point, find nearest center
        points = self.data
        # distances shape => (#points, #centers)
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        # find min distance for each point
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)

        return -sse  # negate to maximize

    # ------------------------------------------------------------
    # (5) Selection: Tournament
    # ------------------------------------------------------------

    def tournament_selection(self, population, fitnesses, k=3):
        """
        Standard tournament selection.
        """
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            tournament = random.sample(zipped, k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    # ------------------------------------------------------------
    # (6) Crossover (Binary or Real)
    # ------------------------------------------------------------

    def crossover(self, p1, p2):
        """
        If binary-coded, do 1-point crossover.
        If real-coded, do a simple blend with random alpha.
        """
        if self.use_binary:
            point = random.randint(1, len(p1) - 1)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            return c1, c2
        else:
            alpha = random.random()
            child1 = tuple(alpha * x1 + (1 - alpha) * x2 for x1, x2 in zip(p1, p2))
            child2 = tuple(alpha * x2 + (1 - alpha) * x1 for x1, x2 in zip(p1, p2))
            return child1, child2

    # ------------------------------------------------------------
    # (7) Mutation (Binary or Real)
    # ------------------------------------------------------------

    def mutation(self, individual, generation, max_gens):
        """
        If binary-coded, do bit-flip mutation with probability = self.mutation_rate * (1 - gen/max_gens).
        If real-coded, do a small random perturbation (also adaptively scaled).
        """
        adaptive_rate = self.mutation_rate * (1.0 - float(generation) / max_gens)

        if self.use_binary:
            ind_list = list(individual)
            for i in range(len(ind_list)):
                if random.random() < adaptive_rate:
                    ind_list[i] = 1 - ind_list[i]  # flip bit
            return ind_list
        else:
            ind_list = list(individual)
            for i in range(len(ind_list)):
                if random.random() < adaptive_rate:
                    shift = random.uniform(-1, 1)  # simple shift
                    ind_list[i] += shift
                    # enforce bounds
                    ind_list[i] = max(
                        min(ind_list[i], self.upper_bound), self.lower_bound
                    )
            return tuple(ind_list)

    # ------------------------------------------------------------
    # (8) SSE Calculation & Plot (for final)
    # ------------------------------------------------------------

    def calculate_sse(self, individual):
        """
        Utility to get the SSE for an individual's cluster centers.
        """
        if self.use_binary:
            centers_vals = self.decode_binary_to_real(individual)
        else:
            centers_vals = individual

        centers = np.array(centers_vals).reshape(self.k, self.dim)
        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)
        return sse

    def plot_clusters(self, best_solution):
        """
        For 2D data only (dim=2):
        plot the data colored by assigned cluster, plus the final cluster centers.
        """
        if self.dim != 2:
            print("plot_clusters() is only implemented for 2D data.")
            return

        if self.use_binary:
            centers_vals = self.decode_binary_to_real(best_solution)
        else:
            centers_vals = best_solution

        centers = np.array(centers_vals).reshape(self.k, 2)
        points = self.data

        # Assign each point to cluster
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        # Plot each cluster
        fig, ax = plt.subplots()
        for cluster_idx in range(self.k):
            cluster_points = points[cluster_assignments == cluster_idx]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, alpha=0.6)

        # Plot centers
        ax.scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X", s=200, edgecolors="k"
        )
        ax.set_title("GA Clustering Result (K=%d)" % self.k)
        plt.show()

    # ------------------------------------------------------------
    # (9) GA Main Loop
    # ------------------------------------------------------------

    def run(self):
        population = self.create_initial_population()

        # We'll track the best SSE (which we want to minimize) -> or track best fitness = -SSE (maximize).
        global_best = None
        global_best_fit = -float("inf")
        global_best_sse = float("inf")

        best_sse_so_far = float("inf")
        no_improvement_counter = 0

        # for table
        try:
            from prettytable import PrettyTable

            table = PrettyTable()
            table.field_names = [
                "Gen",
                "Representation",
                "Best Fitness",
                "Best SSE",
                "Centers (truncated)",
            ]
        except ImportError:
            table = None

        for gen in range(self.max_generations):
            # Evaluate fitness
            fitnesses = [self.fitness_function(ind) for ind in population]

            # Identify best of this generation
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_sse = self.calculate_sse(best_ind)
            self.bestSSEByGeneration.append(current_sse)

            # Update global best
            if current_sse < global_best_sse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_sse = current_sse

            # Early stopping check
            if current_sse < best_sse_so_far - self.min_delta:
                best_sse_so_far = current_sse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen} due to no SSE improvement.")
                break

            # Logging to table
            if table is not None:
                if self.use_binary:
                    real_params = self.decode_binary_to_real(best_ind)
                else:
                    real_params = best_ind
                # Show only first few center coords to avoid giant columns
                truncated_str = (
                    ", ".join([f"{x:.2f}" for x in real_params[:6]]) + " ..."
                )
                table.add_row(
                    [
                        gen,
                        "binary" if self.use_binary else "real",
                        f"{best_fit:.3f}",
                        f"{current_sse:.3f}",
                        truncated_str,
                    ]
                )

            # Selection
            selected = self.tournament_selection(population, fitnesses)

            # Crossover & Mutation -> next gen
            new_pop = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i + 1) % len(selected)]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutation(c1, gen, self.max_generations)
                c2 = self.mutation(c2, gen, self.max_generations)
                new_pop.append(c1)
                new_pop.append(c2)

            # Elitism: preserve best_ind from this generation
            if self.elitism and len(new_pop) > 0:
                new_pop[0] = best_ind

            population = new_pop[: self.pop_size]

        if table is not None:
            print(table)

        # Final check among last population
        final_fitnesses = [self.fitness_function(ind) for ind in population]
        final_best_idx = np.argmax(final_fitnesses)
        final_best_ind = population[final_best_idx]
        final_best_fit = final_fitnesses[final_best_idx]
        final_best_sse = self.calculate_sse(final_best_ind)

        # Compare to global best across all gens
        if global_best is not None and global_best_sse < final_best_sse:
            truly_best = global_best
            truly_best_fit = global_best_fit
            truly_best_sse = global_best_sse
        else:
            truly_best = final_best_ind
            truly_best_fit = final_best_fit
            truly_best_sse = final_best_sse

        # Print final info
        print("\n=== Final Reported Best Clustering (Global) ===")
        if self.use_binary:
            real_params = self.decode_binary_to_real(truly_best)
        else:
            real_params = truly_best
        print(f"Representation: {'binary' if self.use_binary else 'real'}")
        print(f"Best Fitness (=-SSE): {truly_best_fit:.4f}")
        print(f"SSE: {truly_best_sse:.4f}")
        # Optionally print out all center coordinates:
        reshaped_centers = np.array(real_params).reshape(self.k, self.dim)
        print(f"Cluster Centers:\n{reshaped_centers}")

        # Plot final result in 2D
        self.plot_clusters(truly_best)

        return truly_best


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("GA Clustering code demo")

    # Example usage for 2D data, real-coded representation
    ga_cluster = ClusteringGA(
        data=None,  # automatically generate 2D blobs
        k=3,  # number of clusters
        dim=2,
        use_binary_representation=False,
        chrom_length=10,  # unused for real-coded, but keep to match signature
        pop_size=50,
        lower_bound=-10,
        upper_bound=10,
        max_generations=50,
        mutation_rate=0.05,
        patience=8,
        min_delta=0.01,
        elitism=True,
    )
    best_solution = ga_cluster.run()
