#!/usr/bin/env python
import random
import math
import numpy as np
import matplotlib.pyplot as plt


class QuadraticFittingMemeticPBT:
    """
    Memetic + Population-Based Training approach for fitting a quadratic y = a x^2 + b x + c
    to a target function y = a_target x^2 + b_target x + c_target.

    - Memetic: After GA crossover & mutation, we do a local gradient step on each individual.
    - PBT: Each individual also has a 'learning rate' for local search that evolves across the population.
           Periodically, the worst solutions adopt the best solution & hyperparam.
    """

    def __init__(
        self,
        pop_size=20,
        lower_bound=-50,
        upper_bound=50,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        patience=10,
        min_delta=1e-3,
        target_a=1,
        target_b=0,
        target_c=0,
        pbt_interval=5,  # every 5 generations, do a PBT step
    ):
        """
        :param pop_size: population size
        :param lower_bound, upper_bound: range for a,b,c
        :param generations: max generations
        :param mutation_rate: probability for real-coded mutation
        :param crossover_rate: probability of applying crossover
        :param patience: early stopping
        :param min_delta: MSE improvement threshold
        :param target_a,b,c: define the target function
        :param pbt_interval: how often (in gens) to do population-based 'hyperparam copying'
        """
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.patience = patience
        self.min_delta = min_delta

        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

        self.pbt_interval = pbt_interval

    # ----------------------------------------------------------------
    # (A) Representation
    # ----------------------------------------------------------------
    def create_individual(self):
        """
        Individual: (a, b, c, lr)
         - (a,b,c) real-coded in [lower_bound, upper_bound]
         - lr is a small positive learning rate for local search step
        """
        a = random.uniform(self.lower_bound, self.upper_bound)
        b = random.uniform(self.lower_bound, self.upper_bound)
        c = random.uniform(self.lower_bound, self.upper_bound)
        lr = random.uniform(0.001, 0.1)  # local search learning rate
        return (a, b, c, lr)

    def clip_params(self, a, b, c):
        a = max(min(a, self.upper_bound), self.lower_bound)
        b = max(min(b, self.upper_bound), self.lower_bound)
        c = max(min(c, self.upper_bound), self.lower_bound)
        return (a, b, c)

    # ----------------------------------------------------------------
    # (B) Fitness & MSE
    # ----------------------------------------------------------------
    def fitness_function(self, individual):
        """
        We'll keep the 'curviness' approach as before for 'fitness'.
        If a <= 0, big penalty, else measure difference around vertex.
        """
        (a, b, c, lr) = individual
        if a <= 0:
            return -float("inf")
        vertex_x = -b / (2 * a) if abs(a) > 1e-12 else 0
        vertex_y = a * (vertex_x**2) + b * vertex_x + c
        y_left = a * (-1) ** 2 + b * (-1) + c
        y_right = a * (1) ** 2 + b * (1) + c
        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
        return -curviness

    def calculate_mse(self, individual, n_points=100):
        (a, b, c, lr) = individual
        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a * (x_vals**2) + b * x_vals + c
        tgt_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c
        return np.mean((pred_y - tgt_y) ** 2)

    # ----------------------------------------------------------------
    # (C) GA Operators: Selection, Crossover, Mutation
    # ----------------------------------------------------------------
    def tournament_selection(self, population, fitnesses, k=3):
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            group = random.sample(zipped, k)
            winner = max(group, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, p1, p2):
        """
        Real-coded crossover for (a,b,c,lr).
        with probability self.crossover_rate, do a random alpha blend
        """
        if random.random() > self.crossover_rate:
            return p1, p2  # no crossover

        alpha = random.random()
        child1 = []
        child2 = []
        for x1, x2 in zip(p1, p2):
            c1 = alpha * x1 + (1 - alpha) * x2
            c2 = alpha * x2 + (1 - alpha) * x1
            child1.append(c1)
            child2.append(c2)
        return tuple(child1), tuple(child2)

    def mutation(self, individual):
        """
        Real-coded mutation: each gene has probability self.mutation_rate to shift.
        """
        (a, b, c, lr) = individual
        out = [a, b, c, lr]
        for i in range(len(out)):
            if random.random() < self.mutation_rate:
                # random shift
                shift = random.uniform(-1, 1)
                out[i] += shift
        # clip the a,b,c, but allow lr to remain in [0.0001, 1] maybe
        out[0], out[1], out[2] = self.clip_params(out[0], out[1], out[2])
        out[3] = max(0.0001, min(out[3], 1.0))  # learning rate bounds
        return tuple(out)

    # ----------------------------------------------------------------
    # (D) Local Search Step (Memetic)
    # ----------------------------------------------------------------
    def local_search(self, individual):
        """
        Do a single gradient step to reduce MSE w.r.t the target function, using the individual's lr.
        MSE = mean((pred - target)^2). We'll do partial derivatives wrt a,b,c.

        y_pred = a x^2 + b x + c
        derivative wrt a: dMSE/da, etc.

        We pick a small set of x points to approximate the gradient
        """
        (a, b, c, lr) = individual
        # let's pick some random x points or a linspace
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 20)
        grad_a = 0.0
        grad_b = 0.0
        grad_c = 0.0

        for x in x_vals:
            pred = a * (x**2) + b * x + c
            target = self.target_a * (x**2) + self.target_b * x + self.target_c
            diff = pred - target
            # partial wrt a: diff * x^2
            grad_a += diff * (x**2)
            # partial wrt b
            grad_b += diff * x
            # partial wrt c
            grad_c += diff

        # average
        n = len(x_vals)
        grad_a /= n
        grad_b /= n
        grad_c /= n

        # do one step of gradient descent
        # e.g., a' = a - lr * grad_a, etc.
        a_new = a - lr * grad_a
        b_new = b - lr * grad_b
        c_new = c - lr * grad_c

        a_new, b_new, c_new = self.clip_params(a_new, b_new, c_new)
        return (a_new, b_new, c_new, lr)

    # ----------------------------------------------------------------
    # (E) PBT Step
    # ----------------------------------------------------------------
    def population_based_training_step(self, population, fitnesses):
        """
        Sort population by fitness. The bottom portion adopt the top portion's parameters/hyperparams
        Possibly with some random perturbation to lr or partial clone.
        """
        zipped = list(zip(population, fitnesses))
        zipped.sort(key=lambda x: x[1], reverse=True)  # best first
        top_half = zipped[: len(zipped) // 2]
        bottom_half = zipped[len(zipped) // 2 :]

        new_population = []
        # keep top half as is
        for ind_fit in top_half:
            new_population.append(ind_fit[0])
        # for bottom half, copy from random top
        for ind_fit in bottom_half:
            loser = ind_fit[0]
            # randomly pick a winner
            winner = random.choice(top_half)[0]
            # copy winner's (a,b,c,lr)
            new_pop = list(winner)
            # maybe do small tweak to lr
            new_pop[3] *= random.uniform(0.9, 1.1)  # e.g. ±10%
            new_pop[3] = max(0.0001, min(new_pop[3], 1.0))
            new_population.append(tuple(new_pop))

        return new_population

    # ----------------------------------------------------------------
    # (F) Main Loop
    # ----------------------------------------------------------------
    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_performers = []
        best_mse = float("inf")
        no_improvement_counter = 0

        global_best = None
        global_best_fit = -float("inf")
        global_best_mse = float("inf")

        for gen in range(self.generations):
            # evaluate fitness
            fitnesses = [self.fitness_function(ind) for ind in population]

            # do selection
            selected = self.tournament_selection(population, fitnesses)

            # do crossover & mutation
            new_population = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i + 1) % len(selected)]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                new_population.append(c1)
                new_population.append(c2)
            population = new_population[: self.pop_size]

            # do memetic local search on each individual
            memetic_pop = []
            for ind in population:
                improved = self.local_search(ind)
                memetic_pop.append(improved)
            population = memetic_pop

            # evaluate & get best
            fitnesses = [self.fitness_function(ind) for ind in population]
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_mse = self.calculate_mse(best_ind)

            # update global best
            if current_mse < global_best_mse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_mse = current_mse

            # early stopping
            if current_mse < best_mse - self.min_delta:
                best_mse = current_mse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen}")
                break

            best_performers.append((best_ind, best_fit))

            # every pbt_interval, do a population-based training step
            if (gen + 1) % self.pbt_interval == 0:
                fitnesses = [self.fitness_function(ind) for ind in population]
                population = self.population_based_training_step(population, fitnesses)

        # final best from last generation
        final_fits = [self.fitness_function(ind) for ind in population]
        final_idx = np.argmax(final_fits)
        final_best = population[final_idx]
        final_best_fit = final_fits[final_idx]
        final_best_mse = self.calculate_mse(final_best)

        # compare with global best
        if global_best is None:
            truly_best = final_best
            truly_best_fit = final_best_fit
            truly_best_mse = final_best_mse
        else:
            truly_best = global_best
            truly_best_fit = global_best_fit
            truly_best_mse = global_best_mse

        print("\n=== Memetic + PBT Final Report ===")
        print(f"Fitness: {truly_best_fit:.4f}")
        print(f"MSE: {truly_best_mse:.4f}")
        a_b, b_b, c_b, lr_b = truly_best
        print(f"(a,b,c, lr) = ({a_b:.3f}, {b_b:.3f}, {c_b:.3f}, lr={lr_b:.4f})")

        self.plot_evolution(best_performers, truly_best)
        return truly_best

    # ----------------------------------------------------------------
    # (G) Plot
    # ----------------------------------------------------------------
    def plot_evolution(self, best_performers, best_solution):
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 400)
        target_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c

        fig, ax = plt.subplots()
        ax.plot(x_vals, target_y, "k", label="Target", linewidth=2)

        step = max(1, len(best_performers) // 5)
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_performers[::step])))

        for idx, (ind, fit) in enumerate(best_performers[::step]):
            (aa, bb, cc, lr) = ind
            y_vals = aa * (x_vals**2) + bb * x_vals + cc
            ax.plot(
                x_vals, y_vals, color=colors[idx], label=f"Gen {idx*step} fit={fit:.2f}"
            )

        (a_b, b_b, c_b, lr_b) = best_solution
        final_y = a_b * (x_vals**2) + b_b * x_vals + c_b
        ax.plot(x_vals, final_y, "r--", label="Best Final", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Memetic + PBT Quadratic Fitting")
        ax.legend()
        plt.show()


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example run
    mpbt = QuadraticFittingMemeticPBT(
        pop_size=30,
        lower_bound=-10,
        upper_bound=10,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        patience=10,
        min_delta=1e-3,
        target_a=1,  # target = x^2
        target_b=0,
        target_c=0,
        pbt_interval=5,
    )
    best_sol = mpbt.run()
    print("Memetic+PBT best solution:", best_sol)
