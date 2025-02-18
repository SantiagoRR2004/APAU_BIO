#!/usr/bin/env python
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1) Evolution Strategies (ES) with (mu,lambda) and self-adaptive sigmas
# ================================================================


class QuadraticFittingES:
    """
    Evolution Strategies for fitting y = ax^2 + bx + c to a target function.
    Uses (mu, lambda) selection and self-adaptive mutation (one sigma per dimension).
    """

    def __init__(
        self,
        mu=10,  # number of parents
        lambd=40,  # number of offspring
        lower_bound=-50,
        upper_bound=50,
        generations=50,
        patience=10,
        min_delta=1e-3,
        target_a=1,  # target function y = a*x^2 + b*x + c
        target_b=0,
        target_c=0,
        tau=0.5,  # global learning rate for sigmas
        tau_prime=0.2,  # dimension-wise learning rate
    ):
        """
        :param mu: # of parents
        :param lambd: # of offspring
        :param lower_bound, upper_bound: parameter bounds
        :param generations: max generations
        :param patience: early stopping patience
        :param min_delta: MSE improvement threshold
        :param target_a,b,c: define the target function
        :param tau, tau_prime: learning rates for self-adaptive mutation
        """
        self.mu = mu
        self.lambd = lambd
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.generations = generations
        self.patience = patience
        self.min_delta = min_delta

        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

        # self-adaptive parameters
        self.tau = tau
        self.tau_prime = tau_prime

        # dimension = 3 (a, b, c). We'll keep one sigma per dimension => 3
        self.dim = 3

    # ----------------------------------------------------------------
    # (A) Individual Representation
    # ----------------------------------------------------------------
    def create_individual(self):
        """
        Returns (a,b,c, sigmaA, sigmaB, sigmaC)
        where a,b,c are in [lower_bound, upper_bound],
              sigmaX are initial mutation step-sizes (e.g. 1.0).
        """
        a = random.uniform(self.lower_bound, self.upper_bound)
        b = random.uniform(self.lower_bound, self.upper_bound)
        c = random.uniform(self.lower_bound, self.upper_bound)
        sA = 1.0
        sB = 1.0
        sC = 1.0
        return (a, b, c, sA, sB, sC)

    def clip_params(self, a, b, c):
        """
        Ensure a,b,c remain within [lower_bound, upper_bound].
        """
        a = max(min(a, self.upper_bound), self.lower_bound)
        b = max(min(b, self.upper_bound), self.lower_bound)
        c = max(min(c, self.upper_bound), self.lower_bound)
        return (a, b, c)

    # ----------------------------------------------------------------
    # (B) Fitness vs. MSE
    # ----------------------------------------------------------------
    def fitness_function(self, individual):
        """
        We'll do the same 'curviness' approach as your GA code:
         - If a <= 0, heavy penalty
         - measure difference around vertex
        """
        (a, b, c, sA, sB, sC) = individual
        if a <= 0:
            return -float("inf")
        vertex_x = -b / (2 * a) if abs(a) > 1e-12 else 0
        vertex_y = a * (vertex_x**2) + b * vertex_x + c
        y_left = a * (-1) ** 2 + b * (-1) + c
        y_right = a * (1) ** 2 + b * (1) + c
        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
        return -curviness

    def calculate_mse(self, individual, n_points=100):
        (a, b, c, sA, sB, sC) = individual
        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a * (x_vals**2) + b * x_vals + c
        tgt_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c
        mse = np.mean((pred_y - tgt_y) ** 2)
        return mse

    # ----------------------------------------------------------------
    # (C) Main Loop
    # ----------------------------------------------------------------
    def run(self):
        # Initialize parent population
        parents = [self.create_individual() for _ in range(self.mu)]

        best_mse = float("inf")
        no_improvement_counter = 0
        best_performers = []

        # Global best tracking
        global_best = None
        global_best_fit = -float("inf")
        global_best_mse = float("inf")

        for gen in range(self.generations):
            # 1) Generate offspring
            offspring = []
            for _ in range(self.lambd):
                # pick a parent randomly or pick multiple for recombination
                # for basic ES, let's pick 1 random parent => (mu, lambda) style
                p = random.choice(parents)
                (a, b, c, sA, sB, sC) = p

                # 2) Self-adapt the sigmas
                # global factor
                global_factor = math.exp(self.tau * random.gauss(0, 1))
                # dimension factor
                sA *= global_factor * math.exp(self.tau_prime * random.gauss(0, 1))
                sB *= global_factor * math.exp(self.tau_prime * random.gauss(0, 1))
                sC *= global_factor * math.exp(self.tau_prime * random.gauss(0, 1))

                # 3) mutate a,b,c with these sigmas
                a_new = a + sA * random.gauss(0, 1)
                b_new = b + sB * random.gauss(0, 1)
                c_new = c + sC * random.gauss(0, 1)
                a_new, b_new, c_new = self.clip_params(a_new, b_new, c_new)

                child = (a_new, b_new, c_new, sA, sB, sC)
                offspring.append(child)

            # 4) Merge parents + offspring? If (mu+lambda) or only offspring if (mu, lambda).
            # Typically, in a (mu, lambda) strategy, we only pick best mu from the offspring
            # ignoring old parents. So let's do (mu,lambda).
            combined = offspring  # ignoring old parents for selection (mu,lambda)
            # Evaluate combined
            fits = [self.fitness_function(ind) for ind in combined]
            # pick best mu
            sorted_idx = np.argsort(fits)[::-1]  # descending
            parents = [combined[i] for i in sorted_idx[: self.mu]]

            # best in this generation
            best_ind = parents[0]
            best_fit = self.fitness_function(best_ind)
            current_mse = self.calculate_mse(best_ind)

            # update global best
            if current_mse < global_best_mse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_mse = current_mse

            # early stopping check
            if current_mse < best_mse - self.min_delta:
                best_mse = current_mse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen}")
                break

            best_performers.append((best_ind, best_fit))

        # final best among parents (though global best might differ)
        final_fits = [self.fitness_function(p) for p in parents]
        idx_best = np.argmax(final_fits)
        final_best = parents[idx_best]
        final_best_fit = final_fits[idx_best]
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

        print("\n=== (mu,lambda)-ES Final Report ===")
        print(f"Fitness: {truly_best_fit:.4f}")
        print(f"MSE: {truly_best_mse:.4f}")
        a_b, b_b, c_b, sa_b, sb_b, sc_b = truly_best
        print(
            f"(a,b,c) = ({a_b:.3f}, {b_b:.3f}, {c_b:.3f}), sigmas=({sa_b:.3f}, {sb_b:.3f}, {sc_b:.3f})"
        )

        self.plot_evolution(best_performers, truly_best)

        return truly_best

    def plot_evolution(self, best_performers, best_solution):
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 400)
        target_y = self.target_a * x_vals**2 + self.target_b * x_vals + self.target_c

        fig, ax = plt.subplots()
        ax.plot(x_vals, target_y, "k", label="Target", linewidth=2)

        step = max(1, len(best_performers) // 5)
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_performers[::step])))

        for idx, (ind, fit) in enumerate(best_performers[::step]):
            (aa, bb, cc, _, _, _) = ind
            y_vals = aa * (x_vals**2) + bb * x_vals + cc
            ax.plot(
                x_vals, y_vals, color=colors[idx], label=f"Gen {idx*step} fit={fit:.2f}"
            )

        # final best
        (a_b, b_b, c_b, _, _, _) = best_solution
        best_y = a_b * (x_vals**2) + b_b * x_vals + c_b
        ax.plot(x_vals, best_y, "r--", label="Best Final", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("(mu,lambda)-ES Quadratic Fitting")
        ax.legend()
        plt.show()


# ================================================================
# 2) CMA-ES Implementation (Simplified)
# ================================================================


class QuadraticFittingCMAES:
    """
    A simplified CMA-ES approach for the same quadratic fitting task.
    We'll keep a mean + covariance and update them each generation.

    Note: Real, official CMA-ES is quite elaborate. This is a partial
    demonstration of how it might be coded from scratch.
    """

    def __init__(
        self,
        pop_size=20,
        lower_bound=-50,
        upper_bound=50,
        generations=50,
        patience=10,
        min_delta=1e-3,
        target_a=1,
        target_b=0,
        target_c=0,
        sigma_init=5.0,
        learning_rate_cov=0.3,
    ):
        """
        :param pop_size: number of offspring (lambda).
        :param lower_bound, upper_bound: parameter bounds
        :param generations: max generations
        :param patience: early stopping patience
        :param min_delta: MSE improvement threshold
        :param target_a,b,c: define the target function
        :param sigma_init: initial overall step size
        :param learning_rate_cov: how quickly to adapt covariance
        """
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.generations = generations
        self.patience = patience
        self.min_delta = min_delta

        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

        self.dim = 3  # (a,b,c)
        self.sigma_init = sigma_init
        self.lr_cov = learning_rate_cov

    # ----------------------------------------------------------------
    # (A) Fitness & MSE
    # ----------------------------------------------------------------
    def fitness_function(self, params):
        (a, b, c) = params
        if a <= 0:
            return -float("inf")
        # same curviness measure
        vertex_x = -b / (2 * a) if abs(a) > 1e-12 else 0
        vertex_y = a * (vertex_x**2) + b * vertex_x + c
        y_left = a * (-1) ** 2 + b * (-1) + c
        y_right = a * (1) ** 2 + b * (1) + c
        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
        return -curviness

    def calculate_mse(self, params, n_points=100):
        (a, b, c) = params
        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a * (x_vals**2) + b * x_vals + c
        tgt_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c
        mse = np.mean((pred_y - tgt_y) ** 2)
        return mse

    # ----------------------------------------------------------------
    # (B) CMA-ES Setup
    # ----------------------------------------------------------------
    def run(self):
        # mean vector for (a,b,c)
        mean = np.array(
            [
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound),
            ],
            dtype=float,
        )

        # initial covariance
        cov = np.eye(self.dim) * (self.sigma_init**2)

        best_mse = float("inf")
        no_improvement_counter = 0
        best_performers = []

        global_best = None
        global_best_fit = -float("inf")
        global_best_mse = float("inf")

        for gen in range(self.generations):
            # Sample pop_size offspring from N(mean, cov)
            offspring = []
            fits = []
            for _ in range(self.pop_size):
                child = np.random.multivariate_normal(mean, cov)
                # clip to bounds
                for i in range(self.dim):
                    child[i] = max(min(child[i], self.upper_bound), self.lower_bound)
                offspring.append(child)

            # Evaluate
            for ind in offspring:
                f = self.fitness_function(tuple(ind))
                fits.append(f)

            # sort offspring by fitness descending
            sorted_idx = np.argsort(fits)[::-1]
            best_half = sorted_idx[: (self.pop_size // 2)]  # or any fraction
            # pick top solutions for the update
            parents = [offspring[i] for i in best_half]

            best_ind = parents[0]
            best_fit = self.fitness_function(tuple(best_ind))
            current_mse = self.calculate_mse(tuple(best_ind))

            # Update global best
            if current_mse < global_best_mse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_mse = current_mse

            # Early Stopping
            if current_mse < best_mse - self.min_delta:
                best_mse = current_mse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen}")
                break

            best_performers.append((tuple(best_ind), best_fit))

            # CMA-ES style update: recalc mean, then partially update covariance
            # (very simplified)
            new_mean = np.mean(parents, axis=0)

            # Covariance update
            # Typically, we do something like:
            # C <- (1 - lr)*C + lr * (1/k) sum_i (parents[i] - mean)(parents[i]-mean)^T
            # for i in best_half
            # We'll do a simple version
            centered = [p - new_mean for p in parents]
            cov_update = np.zeros_like(cov)
            for c in centered:
                cov_update += np.outer(c, c)
            cov_update /= len(parents)

            cov = (1 - self.lr_cov) * cov + self.lr_cov * cov_update

            # update mean
            mean = new_mean

        # after loop
        if global_best is None:
            # fallback: pick best from last generation
            best_offspring = sorted_idx[0]
            final_best = offspring[best_offspring]
            final_best_fit = fits[best_offspring]
            final_best_mse = self.calculate_mse(tuple(final_best))
            truly_best = final_best
            truly_best_fit = final_best_fit
            truly_best_mse = final_best_mse
        else:
            truly_best = global_best
            truly_best_fit = global_best_fit
            truly_best_mse = global_best_mse

        print("\n=== CMA-ES (Simplified) Final Report ===")
        print(f"Fitness: {truly_best_fit:.4f}")
        print(f"MSE: {truly_best_mse:.4f}")
        a_b, b_b, c_b = truly_best
        print(f"(a,b,c) = ({a_b:.3f}, {b_b:.3f}, {c_b:.3f})")

        self.plot_evolution(best_performers, tuple(truly_best))
        return tuple(truly_best)

    def plot_evolution(self, best_performers, best_solution):
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 400)
        target_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c

        fig, ax = plt.subplots()
        ax.plot(x_vals, target_y, "k", label="Target", linewidth=2)

        step = max(1, len(best_performers) // 5)
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_performers[::step])))

        for idx, (ind, fit) in enumerate(best_performers[::step]):
            a, b, c = ind
            y_vals = a * (x_vals**2) + b * x_vals + c
            ax.plot(
                x_vals, y_vals, color=colors[idx], label=f"Gen {idx*step} fit={fit:.2f}"
            )

        a_b, b_b, c_b = best_solution
        final_y = a_b * (x_vals**2) + b_b * x_vals + c_b
        ax.plot(x_vals, final_y, "r--", label="Best Final", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("CMA-ES Quadratic Fitting (Simplified)")
        ax.legend()
        plt.show()


# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------
if __name__ == "__main__":
    # (A) ES run
    es = QuadraticFittingES(
        mu=10,
        lambd=40,
        lower_bound=-10,
        upper_bound=10,
        generations=40,
        patience=10,
        min_delta=1e-3,
        target_a=1,  # target = x^2
        target_b=0,
        target_c=0,
        tau=0.5,
        tau_prime=0.2,
    )
    best_sol_es = es.run()
    print("ES best solution:", best_sol_es)

    # (B) CMA-ES run
    cmaes = QuadraticFittingCMAES(
        pop_size=20,
        lower_bound=-10,
        upper_bound=10,
        generations=50,
        patience=12,
        min_delta=1e-3,
        target_a=1,  # target = x^2
        target_b=0,
        target_c=0,
        sigma_init=5.0,
        learning_rate_cov=0.3,
    )
    best_sol_cmaes = cmaes.run()
    print("CMA-ES best solution:", best_sol_cmaes)
