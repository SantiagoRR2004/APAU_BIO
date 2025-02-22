#!/usr/bin/env python

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import Clustering


# ---------------------------
#  GP Node Classes
# ---------------------------
class GPNode:
    """
    A node in the GP tree. It can be either:
    - A 'function' node with children (like Add, Sub, Avg),
    - A 'terminal' node storing a random vector in R^(k*dim).

    We assume each node returns a vector of length (k*dim).
    """

    def __init__(
        self, is_function=False, func=None, children=None, terminal_value=None
    ):
        self.is_function = is_function  # True if function node, False if terminal
        self.func = func  # e.g. "add", "sub", "avg" if function
        self.children = children if children else []
        self.terminal_value = terminal_value  # if terminal, a vector in R^(k*dim)

    def evaluate(self):
        """
        Evaluate the node to produce a vector of length (k*dim).
        If terminal, return the stored vector.
        If function, recursively evaluate children and apply the function.
        """
        if not self.is_function:
            # Terminal node => just return the stored vector
            return self.terminal_value
        else:
            # Function node => evaluate children, combine
            child_vals = [child.evaluate() for child in self.children]
            # we assume binary functions here for simplicity
            if len(child_vals) != 2:
                raise ValueError(
                    "This simple example assumes 2 children for function nodes."
                )

            v1, v2 = child_vals
            if self.func == "add":
                return v1 + v2
            elif self.func == "sub":
                return v1 - v2
            elif self.func == "avg":
                return 0.5 * (v1 + v2)
            elif self.func == "mul":
                # element-wise multiplication
                return v1 * v2
            else:
                raise ValueError(f"Unknown function: {self.func}")


class ClusteringGP(Clustering.Clustering):
    """
    Genetic Programming for clustering. Each GP individual is a tree that,
    when evaluated, returns a vector of length (k*dim). Interpreted as K centers in 'dim' space.
    Then we compute SSE of data vs. these centers.
    """

    def __init__(
        self,
        max_depth=4,  # max tree depth for new subtrees
        crossover_rate=0.9,
        mutation_rate=0.3,
        *args,
        **kwargs,
    ):
        """
        :param max_depth: maximum depth for newly generated subtrees
        :param crossover_rate: probability of subtree crossover
        :param mutation_rate: probability of subtree mutation
        """
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Build function set for internal nodes
        # For simplicity, we'll use 2-arity operators
        self.function_set = ["add", "sub", "avg", "mul"]

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------
    # Tree Generation & Mutation
    # ------------------------------------------------------------

    def create_initial_population(self) -> list:
        return [self.create_individual(depth=2) for _ in range(self.pop_size)]

    def create_individual(self, depth=0) -> GPNode:
        """
        Randomly create a tree up to 'max_depth'. We'll mix function and terminal nodes.
        """
        # If we've reached max_depth or at the top (and random chance), make a terminal
        if (depth >= self.max_depth) or (depth > 0 and random.random() < 0.3):
            return self._random_terminal_node()
        else:
            # make a function node
            func = random.choice(self.function_set)
            left_child = self.create_individual(depth + 1)
            right_child = self.create_individual(depth + 1)
            node = GPNode(
                is_function=True, func=func, children=[left_child, right_child]
            )
            return node

    def _random_terminal_node(self):
        """
        Create a terminal node that stores a random vector in R^(k*dim).
        """
        vec = np.random.uniform(
            self.lower_bound, self.upper_bound, size=self.vector_size
        )
        return GPNode(is_function=False, terminal_value=vec)

    def _subtree_mutation(self, node, current_depth=0):
        """
        With some probability, replace this node (and below) with a new random subtree.
        Otherwise recurse into its children if it's a function node.
        """
        if random.random() < self.mutation_rate:
            return self.create_individual(depth=current_depth)
        else:
            if node.is_function:
                node.children[0] = self._subtree_mutation(
                    node.children[0], current_depth + 1
                )
                node.children[1] = self._subtree_mutation(
                    node.children[1], current_depth + 1
                )
            return node

    # ------------------------------------------------------------
    # Crossover (Subtree)
    # ------------------------------------------------------------
    def _subtree_crossover(self, parent1, parent2):
        """
        Koza-style subtree crossover: pick a random node in parent1,
        swap it with a random node in parent2.
        """
        # Convert each tree to a list of nodes for random selection
        p1_nodes = self._collect_nodes(parent1)
        p2_nodes = self._collect_nodes(parent2)

        # choose random node from each
        node1 = random.choice(p1_nodes)
        node2 = random.choice(p2_nodes)

        # swap them (shallow)
        node1.is_function, node2.is_function = node2.is_function, node1.is_function
        node1.func, node2.func = node2.func, node1.func
        node1.terminal_value, node2.terminal_value = (
            node2.terminal_value,
            node1.terminal_value,
        )
        node1.children, node2.children = node2.children, node1.children

        return parent1, parent2

    def _collect_nodes(self, root):
        """
        Collect all nodes in the subtree for random selection (preorder).
        """
        nodes = [root]
        if root.is_function:
            nodes += self._collect_nodes(root.children[0])
            nodes += self._collect_nodes(root.children[1])
        return nodes

    # ------------------------------------------------------------
    # Fitness & SSE
    # ------------------------------------------------------------
    def _fitness(self, root):
        """
        Evaluate the tree -> produce cluster centers in R^(k*dim).
        Then compute SSE for data vs. these centers.
        Return fitness = -SSE.
        """
        vec = root.evaluate()  # shape (k*dim,)
        # clip to keep them in [lower_bound, upper_bound], optional
        vec = np.clip(vec, self.lower_bound, self.upper_bound)
        centers = vec.reshape(self.k, self.dim)

        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists**2)
        return -sse

    def _calculate_sse(self, root):
        """
        Just the SSE (positive value).
        """
        fit = self._fitness(root)
        return -fit

    # ------------------------------------------------------------
    # Selection: Tournament
    # ------------------------------------------------------------
    def _tournament_selection(self, population, fitnesses, tsize=3):
        """
        Return one winner from a random subset of size tsize.
        """
        contenders = random.sample(list(zip(population, fitnesses)), tsize)
        winner = max(contenders, key=lambda x: x[1])[0]
        return winner

    # ------------------------------------------------------------
    # Main GP Loop
    # ------------------------------------------------------------
    def run(self):
        best_sse = float("inf")
        no_improve_count = 0

        # Evaluate initial population
        fitnesses = [self._fitness(ind) for ind in self.population]

        # Keep track of global best
        global_best = None
        global_best_fit = -float("inf")

        for gen in range(self.max_generations):
            new_population = []

            # Elitism: keep best from current generation
            best_idx = np.argmax(fitnesses)
            best_ind = self.population[best_idx]
            best_fit = fitnesses[best_idx]
            gen_sse = -best_fit
            self.bestSSEByGeneration.append(gen_sse)

            if gen_sse < best_sse:
                best_sse = gen_sse
                no_improve_count = 0
            else:
                no_improve_count += 1

            if best_fit > global_best_fit:
                global_best = self._clone_tree(best_ind)
                global_best_fit = best_fit

            if no_improve_count >= self.patience:
                print(f"Early stopping at generation {gen} (no SSE improvement).")
                break

            # Logging
            print(f"Gen {gen}, Best SSE = {gen_sse:.4f}")

            # Create new population
            while len(new_population) < self.pop_size:
                # Selection
                p1 = self._tournament_selection(self.population, fitnesses, tsize=3)
                p2 = self._tournament_selection(self.population, fitnesses, tsize=3)

                # Crossover
                if random.random() < self.crossover_rate:
                    c1 = self._clone_tree(p1)
                    c2 = self._clone_tree(p2)
                    c1, c2 = self._subtree_crossover(c1, c2)
                else:
                    c1 = self._clone_tree(p1)
                    c2 = self._clone_tree(p2)

                # Mutation
                c1 = self._subtree_mutation(c1)
                c2 = self._subtree_mutation(c2)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            # Replace population
            self.population = new_population
            fitnesses = [self._fitness(ind) for ind in self.population]

        # Final check among last population
        final_fitnesses = [self._fitness(ind) for ind in self.population]
        final_best_idx = np.argmax(final_fitnesses)
        final_best_ind = self.population[final_best_idx]
        final_best_fit = final_fitnesses[final_best_idx]

        if final_best_fit > global_best_fit:
            truly_best = final_best_ind
            truly_best_fit = final_best_fit
        else:
            truly_best = global_best
            truly_best_fit = global_best_fit

        best_sse_final = -truly_best_fit

        print("\n=== Final GP Clustering ===")
        print(f"Best Fitness (=-SSE): {truly_best_fit:.4f}")
        print(f"SSE: {best_sse_final:.4f}")

        # self.plot_clusters(truly_best)
        return truly_best

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _clone_tree(self, root):
        """
        Deep copy of a tree node.
        """
        new_node = GPNode(
            is_function=root.is_function,
            func=root.func,
            terminal_value=None if root.is_function else np.copy(root.terminal_value),
        )
        if root.is_function:
            new_node.children = [self._clone_tree(ch) for ch in root.children]
        return new_node

    def plot_clusters(self, root):
        """
        Plot the final clusters if dim=2
        """
        if self.dim != 2:
            print("plot_clusters() only supports dim=2.")
            return

        vec = root.evaluate()  # get cluster center vector
        vec = np.clip(vec, self.lower_bound, self.upper_bound)
        centers = vec.reshape(self.k, self.dim)
        dists = np.linalg.norm(self.data[:, None, :] - centers[None, :, :], axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        plt.figure()
        for c_idx in range(self.k):
            pts = self.data[cluster_assignments == c_idx]
            plt.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.6)
        plt.scatter(
            centers[:, 0], centers[:, 1], c="red", marker="X", s=200, edgecolors="k"
        )
        plt.title("GP-based Clustering (Tree Representation)")
        plt.show()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== GP for Clustering Demo ===")

    gp_cluster = ClusteringGP(
        data=None,  # generate random 2D blobs
        k=3,
        dim=2,
        pop_size=30,
        max_depth=4,
        lower_bound=-10,
        upper_bound=10,
        max_generations=30,
        crossover_rate=0.9,
        mutation_rate=0.3,
        patience=5,
        min_delta=1e-3,
        seed=42,
    )
    best_solution = gp_cluster.run()
