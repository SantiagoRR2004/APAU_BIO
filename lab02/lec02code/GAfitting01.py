#!/usr/bin/env python

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class QuadraticFittingGA:
    """
    Genetic Algorithm for fitting a quadratic function y = a x^2 + b x + c.
    Can use either real-coded or binary-coded representation.
    """

    def __init__(
        self,
        use_binary_representation=False,
        chrom_length=30,
        pop_size=50,
        lower_bound=-50,
        upper_bound=50,
        generations=50,
        mutation_rate=0.05,
        patience=10,
        min_delta=0.001,
        target_a=1,         # Target function's 'a'
        target_b=0,         # Target function's 'b'
        target_c=0,         # Target function's 'c'
        elitism=True
    ):
        """
        :param use_binary_representation: If True, we use binary-coded. Otherwise, real-coded representation.
        :param chrom_length: length of each gene if using binary-coded representation (applies to each param a,b,c).
        :param pop_size: population size
        :param lower_bound: min possible param value
        :param upper_bound: max possible param value
        :param generations: max number of generations
        :param mutation_rate: base mutation probability
        :param patience: early-stopping patience based on MSE
        :param min_delta: MSE improvement threshold
        :param target_a, target_b, target_c: define the target function y = a x^2 + b x + c
        :param elitism: keep the best individual from current gen into next gen
        """

        self.use_binary = use_binary_representation
        self.chrom_length = chrom_length   # per parameter if using binary
        self.params_per_ind = 3           # (a, b, c)
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.min_delta = min_delta
        self.elitism = elitism

        # For the target function y = a x^2 + b x + c
        self.target_a = target_a
        self.target_b = target_b
        self.target_c = target_c

        # A single chromosome length if using binary will be: self.params_per_ind * self.chrom_length
        # e.g., if chrom_length=10, total bits = 3*10=30

    # ------------------------------------------------------------
    # (1) GA Representation
    # ------------------------------------------------------------

    def encode_real_to_binary(self, real_values):
        """
        Convert (a,b,c) real-coded tuple into a single bitstring list (binary-coded).
        We'll divide the bitstring into 3 segments, each of length self.chrom_length.
        """
        bitstring = []
        for val in real_values:
            bits = self._float_to_bits(val)
            bitstring.extend(bits)
        return bitstring

    def decode_binary_to_real(self, bitstring):
        """
        Convert a bitstring (list of 0/1) back to a tuple (a,b,c) in real domain.
        Splits bitstring into 3 chunks of length = self.chrom_length.
        """
        chunk_size = self.chrom_length
        real_values = []
        for i in range(self.params_per_ind):
            chunk = bitstring[i*chunk_size : (i+1)*chunk_size]
            val = self._bits_to_float(chunk)
            real_values.append(val)
        return tuple(real_values)

    def _float_to_bits(self, val):
        """
        Map float val in [lower_bound, upper_bound] -> a chunk of length self.chrom_length bits.
        """
        # ensure val is clipped
        val = max(min(val, self.upper_bound), self.lower_bound)
        max_bits_val = (1 << self.chrom_length) - 1  # 2^self.chrom_length - 1
        # Normalize in [0,1]
        norm = (val - self.lower_bound) / (self.upper_bound - self.lower_bound)
        # Map to integer in [0, max_bits_val]
        int_val = int(round(norm * max_bits_val))
        # Convert int_val to bit list
        bits = [(int_val >> i) & 1 for i in range(self.chrom_length)]
        # The above is LSB->MSB; reverse to get standard ordering
        bits.reverse()
        return bits

    def _bits_to_float(self, bits):
        """
        Convert chunk of bits -> float in [lower_bound, upper_bound].
        bits[0] is MSB, bits[-1] is LSB if reversed above.
        """
        bitstring = bits[:]
        # Convert list to integer
        val_int = 0
        for b in bitstring:
            val_int = (val_int << 1) | b
        max_bits_val = (1 << self.chrom_length) - 1
        # Convert back from int to float in [lower_bound, upper_bound]
        norm = val_int / max_bits_val
        real_val = self.lower_bound + norm * (self.upper_bound - self.lower_bound)
        return real_val

    def create_individual(self):
        """
        Create a single individual.
        If self.use_binary, returns a bitstring.
        If real-coded, returns a tuple (a,b,c).
        """
        if self.use_binary:
            # internally create random (a,b,c) in real domain, then encode
            real_vals = (
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound)
            )
            return self.encode_real_to_binary(real_vals)
        else:
            # real-coded
            return (
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound),
                random.uniform(self.lower_bound, self.upper_bound)
            )

    # ------------------------------------------------------------
    # (2) Fitness Function
    # ------------------------------------------------------------

    def fitness_function(self, individual):
        """
        For the given individual, compute 'fitness'.
        We'll adapt your example: we want an upright parabola => a>0,
        and we minimize curviness around x=-1, x=1, etc.
        """
        # decode if binary-coded
        if self.use_binary:
            real_params = self.decode_binary_to_real(individual)
        else:
            real_params = individual

        a, b, c = real_params
        if a <= 0:
            return -float('inf')  # severely penalize downward-facing

        # Vertex of parabola
        vertex_x = -b / (2*a) if abs(a) > 1e-12 else 0
        vertex_y = a*(vertex_x**2) + b*vertex_x + c

        # Evaluate curvature measure
        y_left  = a*(-1)**2 + b*(-1) + c
        y_right = a*( 1)**2 + b*( 1) + c
        curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)

        return -curviness  # negative => want to minimize curviness

    # ------------------------------------------------------------
    # (3) Initialization & Population
    # ------------------------------------------------------------

    def create_initial_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(self.create_individual())
        return population

    # ------------------------------------------------------------
    # (4) Selection Operators
    # ------------------------------------------------------------

    def tournament_selection(self, population, fitnesses, k=3):
        """
        standard tournament selection
        """
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            tournament = random.sample(zipped, k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    # ------------------------------------------------------------
    # (5) Crossover Operators (Binary or Real)
    # ------------------------------------------------------------

    def crossover(self, p1, p2):
        """
        If binary-coded, do 1-point crossover.
        If real-coded, do the 'alpha' blend crossover.
        """
        if self.use_binary:
            point = random.randint(1, len(p1)-1)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            return c1, c2
        else:
            # Real-coded crossover with random alpha
            alpha = random.random()
            child1 = tuple(alpha*x1 + (1-alpha)*x2 for x1,x2 in zip(p1,p2))
            child2 = tuple(alpha*x2 + (1-alpha)*x1 for x1,x2 in zip(p1,p2))
            return child1, child2

    # ------------------------------------------------------------
    # (6) Mutation Operators (Binary or Real)
    # ------------------------------------------------------------

    def mutation(self, individual, generation, max_gens):
        """
        If binary-coded, do bit-flip mutation with probability = self.mutation_rate * (1 - generation/max_gens)
        If real-coded, do random small perturbation
        """
        adaptive_rate = self.mutation_rate * (1.0 - float(generation)/max_gens)

        if self.use_binary:
            ind_list = list(individual)
            for i in range(len(ind_list)):
                if random.random() < adaptive_rate:
                    ind_list[i] = 1 - ind_list[i]  # flip bit
            return ind_list
        else:
            # real-coded
            ind_list = list(individual)
            for i in range(len(ind_list)):
                if random.random() < adaptive_rate:
                    # random small shift
                    shift = random.uniform(-1, 1)
                    ind_list[i] += shift
                    # enforce bounds
                    ind_list[i] = max(min(ind_list[i], self.upper_bound), self.lower_bound)
            return tuple(ind_list)

    # ------------------------------------------------------------
    # (7) MSE Calculation & Plotting
    # ------------------------------------------------------------

    def calculate_mse(self, individual, n_points=100):
        """
        Compare individual's parabola to the target function (self.target_a, self.target_b, self.target_c)
        """
        if self.use_binary:
            a,b,c = self.decode_binary_to_real(individual)
        else:
            a,b,c = individual

        x_vals = np.linspace(self.lower_bound, self.upper_bound, n_points)
        pred_y = a*(x_vals**2) + b*x_vals + c
        tgt_y  = self.target_a*(x_vals**2) + self.target_b*x_vals + self.target_c
        mse = np.mean((pred_y - tgt_y)**2)
        return mse

    def plot_evolution(self, best_performers, best_solution):
        """
        Plot the final best solution vs. the target function + some intermediate solutions.
        """
        fig, ax = plt.subplots()
        # plot target
        x_vals = np.linspace(self.lower_bound, self.upper_bound, 400)
        target_y = self.target_a*x_vals**2 + self.target_b*x_vals + self.target_c
        ax.plot(x_vals, target_y, 'k', label="Target", linewidth=2)

        # step through generations
        step = max(1, len(best_performers)//5)
        colors = plt.cm.viridis(np.linspace(0,1,len(best_performers[::step])))
        for idx,(ind,fit) in enumerate(best_performers[::step]):
            if self.use_binary:
                a,b,c = self.decode_binary_to_real(ind)
            else:
                a,b,c = ind
            y_vals = a*x_vals**2 + b*x_vals + c
            ax.plot(x_vals, y_vals, color=colors[idx], label=f"Gen {idx*step} fit={fit:.2f}")

        # final best in red dotted
        """
        if self.use_binary:
            a_b,c_b = self.decode_binary_to_real(best_solution)
            a_b, b_b, c_b = a_b,c_b
        else:
            a_b, b_b, c_b = best_solution
        """

        if self.use_binary:
            a_b, b_b, c_b = self.decode_binary_to_real(best_solution)
        else:
            a_b, b_b, c_b = best_solution



        best_y = a_b*x_vals**2 + b_b*x_vals + c_b
        ax.plot(x_vals, best_y, 'r--', label="Best Final", linewidth=2)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("GA Evolution of Quadratic Solutions")
        ax.legend()
        plt.show()


    # ------------------------------------------------------------
    # (8) GA Main Loop (with Global Best Tracking)
    # ------------------------------------------------------------

    def run(self):
        # create population
        population = self.create_initial_population()
        best_performers = []

        # We'll keep a "global best" to avoid losing good solutions
        global_best = None
        global_best_fit = -float('inf')   # because we are maximizing fitness
        global_best_mse = float('inf')

        best_mse = float('inf')
        no_improvement_counter = 0

        # for table
        try:
            from prettytable import PrettyTable
            table = PrettyTable()
            table.field_names = ["Gen", "Representation", "Fitness", "MSE", "params(a,b,c)"]
        except ImportError:
            table = None

        for gen in range(self.generations):
            # evaluate fitness
            fitnesses = []
            for ind in population:
                f = self.fitness_function(ind)
                fitnesses.append(f)

            # best of this generation
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]
            current_mse = self.calculate_mse(best_ind)

            # -- Update "global best" if this generation's best is better (in terms of MSE or fitness) --
            # Here, we rely on MSE improvement because your fitness function might not reflect the actual error
            if current_mse < global_best_mse:
                global_best = best_ind
                global_best_fit = best_fit
                global_best_mse = current_mse

            # check improvement for early stopping (based on MSE)
            if current_mse < best_mse - self.min_delta:
                best_mse = current_mse
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                print(f"Early stopping at generation {gen} due to no MSE improvement.")
                break

            best_performers.append((best_ind, best_fit))

            # Logging to table
            if table is not None:
                if self.use_binary:
                    real_params = self.decode_binary_to_real(best_ind)
                else:
                    real_params = best_ind
                table.add_row([gen,
                            "binary" if self.use_binary else "real",
                            round(best_fit,3),
                            round(current_mse,5),
                            f"{round(real_params[0],3)}, {round(real_params[1],3)}, {round(real_params[2],3)}"])

            # selection
            selected = self.tournament_selection(population, fitnesses)

            # crossover & mutation -> next population
            new_pop = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(selected)]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutation(c1, gen, self.generations)
                c2 = self.mutation(c2, gen, self.generations)
                new_pop.append(c1)
                new_pop.append(c2)

            # elitism: preserve best_ind from *this generation* if desired
            if self.elitism:
                new_pop[0] = best_ind

            population = new_pop[:self.pop_size]

        # After we finish all generations (or early stopping):
        if table is not None:
            print(table)

        # final best from last generation
        final_fitnesses = [self.fitness_function(ind) for ind in population]
        final_best = population[np.argmax(final_fitnesses)]
        final_best_fit = max(final_fitnesses)
        final_best_mse = self.calculate_mse(final_best)

        # but we have a "global best" that might be better
        # let's define "truly best" as the global one
        truly_best = global_best if global_best is not None else final_best
        truly_best_fit = global_best_fit if global_best is not None else final_best_fit
        truly_best_mse = global_best_mse if global_best is not None else final_best_mse

        # print final info
        print("\n=== Final Reported Best Solution (Global) ===")
        if self.use_binary:
            real_params = self.decode_binary_to_real(truly_best)
        else:
            real_params = truly_best

        print(f"Representation: {'binary' if self.use_binary else 'real-coded'}")
        print(f"Fitness: {truly_best_fit:.4f}")
        print(f"MSE: {truly_best_mse:.4f}")
        print(f"Params (a,b,c) = {real_params}")

        # plot evolution w.r.t the final "global best" rather than last gen best
        self.plot_evolution(best_performers, truly_best)

        return truly_best

    

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("GA code")
    real_mode=True


    
    if real_mode:

        print("# 1) Real-coded version")
        ga_real = QuadraticFittingGA(
            use_binary_representation=False,
            chrom_length=20,  # not used in real-coded
            pop_size=80,
            lower_bound=-10,
            upper_bound=10,
            generations=50,
            mutation_rate=0.05,
            patience=10,
            min_delta=1e-3,
            target_a=1,        # target function = x^2
            target_b=0,
            target_c=0,
            elitism=True
        )
        best_sol_real = ga_real.run()

    else:
        print("# 2) Binary-coded version")
        ga_binary = QuadraticFittingGA(
            use_binary_representation=True,
            chrom_length=10,   # each param has 10 bits, total 30 bits
            pop_size=60,
            lower_bound=-10,
            upper_bound=10,
            generations=40,
            mutation_rate=0.02,
            patience=8,
            min_delta=1e-3,
            target_a=1,        # target function = x^2
            target_b=0,
            target_c=0,
            elitism=True
        )
        best_sol_binary = ga_binary.run()
