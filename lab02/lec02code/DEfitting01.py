#!/usr/bin/env python
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class QuadraticFittingDE:
    """
    Differential Evolution for fitting a quadratic function y = a x^2 + b x + c,
    to a target function y = target_a x^2 + target_b x + target_c.

    We keep a 'fitness_function' that tries to shape an upright parabola with minimal
    'curviness' (like your GA approach), plus an MSE measure for early stopping.
    """

    def __init__(
        self,
        pop_size=50,
        lower_bound=-50,
        upper_bound=50,
        generations=50,
        F=0.8,                # DE differential weight
        CR=0.9,               # Crossover probability
        mutation_strategy="rand/1/bin",
        patience=10,
        min_delta=1e-3,
        target_a=1,           # target function y = a*x^2 + b*x + c
        target_b=0,
        target_c=0
    ):
        """
        :param pop_size: population size
        :param lower_bound: min param (a,b,c)
        :param upper_bound: max param (a,b,c)
        :param generations: max generations
        :param F: differential weight
        :param CR: crossover probability
        :param mutation_strategy: e.g. 'rand/1/bin'
        :param patience: early stopping patience
        :param min_delta: min improvement in MSE to reset patience
        :param target_a,b,c: define the target function
        """
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.generations = generations
        self.F = F
        self.CR = CR
        self.strategy = mutation_strategy

        self.patience = patience
        self.min_delta = min_delta

        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

    # ----------------------------------------------------------------
    # (1) Population initialization
    # ----------------------------------------------------------------
    def create_initial_population(self):
        """
        Each individual is (a,b,c) in real-coded form within [lower_bound, upper_bound].
        """
        pop = []
        for _ in range(self.pop_size):
            a = random.uniform(self.lower_bound, self.upper_bound)
            b = random.uniform(self.lower_bound, self.upper_bound)
            c = random.uniform(self.lower_bound, self.upper_bound)
            pop.append((a,b,c))
        return pop

    # ----------------------------------------------------------------
    # (2) Fitness function (similar to your GA code)
    # ----------------------------------------------------------------
    def fitness_function(self, individual):
        """
        If 'a' <= 0, penalize heavily. Then measure 'curviness'.
        Negative of curviness => higher fitness means flatter top near x=-1,1.

        This is not strictly the MSE to the target function. 
        But we can still use MSE as a separate measure for early stopping/plotting.
        """
        (a,b,c) = individual
        if a <= 0:
            return -float('inf')  # penalize downward parabolas

        # Vertex
        vertex_x = -b/(2*a) if abs(a) > 1e-12 else 0
        vertex_y = a*(vertex_x**2) + b*vertex_x + c
        y_left  = a*(-1)**2 + b*(-1) + c
        y_right = a*(1)**2 + b*(1) + c

        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
        return -curviness  # we want smaller curviness => higher fitness

    # ----------------------------------------------------------------
    # (3) MSE measure to the target function
    # ----------------------------------------------------------------
    def calculate_mse(self, individual, n_points=100):
        """
        Compare individual's parabola to target (target_a, target_b, target_c).
        We'll do this for early stopping or final checks.
        """
        (a,b,c) = individual
        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a*(x_vals**2) + b*x_vals + c
        tgt_y = self.target_a*(x_vals**2) + self.target_b*x_vals + self.target_c
        mse = np.mean((pred_y - tgt_y)**2)
        return mse

    # ----------------------------------------------------------------
    # (4) DE Operators: Mutation & Crossover
    # ----------------------------------------------------------------
    def mutate(self, pop, current_idx):
        """
        'rand/1/bin' strategy: pick r1, r2, r3 distinct from current_idx
        v = pop[r1] + F*(pop[r2] - pop[r3])
        """
        # pick distinct indices
        indices = list(range(self.pop_size))
        indices.remove(current_idx)
        r1, r2, r3 = random.sample(indices, 3)

        x_r1 = pop[r1]
        x_r2 = pop[r2]
        x_r3 = pop[r3]

        # v_i = x_r1 + F*(x_r2 - x_r3)
        v = []
        for i in range(3):  # 3 params: a,b,c
            mutated_val = x_r1[i] + self.F*(x_r2[i] - x_r3[i])
            # keep in bounds
            mutated_val = max(min(mutated_val, self.upper_bound), self.lower_bound)
            v.append(mutated_val)

        return tuple(v)

    def crossover(self, target_vec, donor_vec):
        """
        Binomial crossover: for each param, pick from donor_vec with prob CR,
        else from target_vec.
        """
        trial = []
        for i in range(3):
            if random.random() < self.CR:
                trial.append(donor_vec[i])
            else:
                trial.append(target_vec[i])
        return tuple(trial)

    # ----------------------------------------------------------------
    # (5) Run the DE main loop
    # ----------------------------------------------------------------
    def run(self):
        population = self.create_initial_population()
        best_performers = []

        global_best = None
        global_best_fit = -float('inf')
        global_best_mse = float('inf')

        best_mse_so_far = float('inf')
        no_improvement_counter = 0

        # try to log via prettytable
        try:
            from prettytable import PrettyTable
            table = PrettyTable()
            table.field_names = ["Gen", "Best_Fit", "MSE", "(a,b,c)"]
        except ImportError:
            table = None

        for gen in range(self.generations):
            new_population = []

            # For each individual i in population:
            for i in range(self.pop_size):
                target_vec = population[i]
                # Mutation -> donor vector
                donor_vec = self.mutate(population, i)
                # Crossover -> trial vector
                trial_vec = self.crossover(target_vec, donor_vec)

                # Evaluate
                trial_fit = self.fitness_function(trial_vec)
                target_fit = self.fitness_function(target_vec)

                # Selection -> pick better
                if trial_fit > target_fit:
                    new_population.append(trial_vec)
                else:
                    new_population.append(target_vec)

            population = new_population

            # Evaluate population
            fitnesses = [self.fitness_function(ind) for ind in population]
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_mse = self.calculate_mse(best_ind)

            # Update global best if improved in MSE
            if current_mse < global_best_mse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_mse = current_mse

            # Check improvement for early stopping
            if current_mse < best_mse_so_far - self.min_delta:
                best_mse_so_far = current_mse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen} (no MSE improvement).")
                break

            # store for plotting
            best_performers.append((best_ind, best_fit))

            # table row
            if table is not None:
                a,b,c = best_ind
                table.add_row([
                    gen,
                    f"{best_fit:.4f}",
                    f"{current_mse:.4f}",
                    f"({a:.3f}, {b:.3f}, {c:.3f})"
                ])

        # done
        if table is not None:
            print(table)

        # final best from last generation
        final_fit_list = [self.fitness_function(ind) for ind in population]
        final_best_idx = np.argmax(final_fit_list)
        final_best = population[final_best_idx]
        final_best_fit = final_fit_list[final_best_idx]
        final_best_mse = self.calculate_mse(final_best)

        # Compare final best vs global best
        truly_best = global_best
        truly_best_fit = global_best_fit
        truly_best_mse = global_best_mse

        print("\n=== DE Final Reported Best Solution (Global) ===")
        print(f"Fitness: {truly_best_fit:.4f}")
        print(f"MSE: {truly_best_mse:.4f}")
        print(f"Params (a,b,c) = {truly_best}")

        self.plot_evolution(best_performers, truly_best)
        return truly_best

    # ----------------------------------------------------------------
    # (6) Plot Evolution
    # ----------------------------------------------------------------
    def plot_evolution(self, best_performers, best_solution):
        """
        Similar to your GA's plot, show how the best solution evolves vs. the target.
        best_performers is a list of (individual, fitness).
        best_solution is the final global best.
        """
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 400)
        target_y = self.target_a*x_vals**2 + self.target_b*x_vals + self.target_c

        fig, ax = plt.subplots()
        ax.plot(x_vals, target_y, 'k', label="Target", linewidth=2)

        step = max(1, len(best_performers)//5)
        colors = plt.cm.viridis(np.linspace(0,1,len(best_performers[::step])))
        for idx, (ind, fit) in enumerate(best_performers[::step]):
            a,b,c = ind
            y_vals = a*(x_vals**2) + b*x_vals + c
            ax.plot(x_vals, y_vals, color=colors[idx], label=f"Gen {idx*step} fit={fit:.2f}")

        # final best in red dotted
        a_b, b_b, c_b = best_solution
        final_y = a_b*(x_vals**2) + b_b*x_vals + c_b
        ax.plot(x_vals, final_y, 'r--', label="Best Final", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Differential Evolution for Quadratic Fitting")
        ax.legend()
        plt.show()


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    de = QuadraticFittingDE(
        pop_size=60,
        lower_bound=-10,
        upper_bound=10,
        generations=50,
        F=0.8,
        CR=0.9,
        patience=10,
        min_delta=1e-3,
        target_a=1,  # target = x^2
        target_b=0,
        target_c=0
    )
    best_sol = de.run()
    print("DE best solution:", best_sol)
