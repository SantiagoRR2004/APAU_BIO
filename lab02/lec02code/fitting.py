from abc import ABC, abstractmethod
import random
import numpy as np
import matplotlib.pyplot as plt


class AbstractFitting(ABC):
    def __init__(
        self,
        mu: int = 10,  # number of parents
        lower_bound: int = -50,
        upper_bound: int = 50,
        target_a: int = 1,
        target_b: int = 0,
        target_c: int = 0,
    ):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

    def create_individual(self):
        """
        Create a random individual (a, b, c) in the range [lower_bound, upper_bound]
        """
        a = random.uniform(self.lower_bound, self.upper_bound)
        b = random.uniform(self.lower_bound, self.upper_bound)
        c = random.uniform(self.lower_bound, self.upper_bound)
        return [a, b, c]

    def create_population(self):
        """
        Create a population of random individuals
        """
        return [self.create_individual() for _ in range(self.mu)]

    def fitness_function(self, individual):
        """
        If 'a' <= 0, penalize heavily. Then measure 'curviness'.
        Negative of curviness => higher fitness means flatter top near x=-1,1.

        This is not strictly the MSE to the target function.
        But we can still use MSE as a separate measure for early stopping/plotting.
        """
        (a, b, c) = individual[:3]
        if a <= 0:
            return -float("inf")  # penalize downward parabolas

        # Vertex
        vertex_x = -b / (2 * a) if abs(a) > 1e-12 else 0
        vertex_y = a * (vertex_x**2) + b * vertex_x + c
        y_left = a * (-1) ** 2 + b * (-1) + c
        y_right = a * (1) ** 2 + b * (1) + c

        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
        return -curviness  # we want smaller curviness => higher fitness

    @abstractmethod
    def calculate_mse(self, individual, n_point=100):
        """
        Compare individual's parabola to the target function (self.target_a, self.target_b, self.target_c)
        """
        a, b, c = individual[:3]

        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a * (x_vals**2) + b * x_vals + c
        tgt_y = self.target_a * (x_vals**2) + self.target_b * x_vals + self.target_c
        mse = np.mean((pred_y - tgt_y) ** 2)
        return mse

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot_evolution(self):
        pass
        # plt.savefig(f"{self.__class__.__name__}.png")
