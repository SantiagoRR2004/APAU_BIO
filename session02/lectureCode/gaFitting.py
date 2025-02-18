import numpy as np
import random


# Objective function (to be maximized)
def objective_function(x):
    return -(x**2 - 10 * np.cos(2 * np.pi * x) + 10)


# Initialize population with real values in the range [-5.12, 5.12]
def initialize_population(size, bounds):
    return np.random.uniform(bounds[0], bounds[1], size)


# Evaluate fitness (objective function value)
def evaluate_fitness(population):
    return np.array([objective_function(x) for x in population])


# Tournament selection
def tournament_selection(population, fitness, k=3):
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda i: fitness[i])
    return population[best_index]


# Simulated Binary Crossover (SBX)
def simulated_binary_crossover(parent1, parent2, eta=2):
    if np.random.rand() > 0.9:  # Small chance of crossover
        return parent1, parent2

    u = np.random.rand()
    beta = (
        (2 * u) ** (1 / (eta + 1))
        if u <= 0.5
        else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )

    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    return np.clip(child1, -5.12, 5.12), np.clip(child2, -5.12, 5.12)


# Gaussian mutation
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):
    if np.random.rand() < mutation_rate:
        return np.clip(individual + np.random.normal(0, sigma), -5.12, 5.12)
    return individual


# Replacement strategy: Elitism + Offspring replacement
def replacement(population, offspring, fitness, elitism_k=2):
    sorted_indices = np.argsort(fitness)[::-1]  # Sort by fitness (descending)
    elite = [
        population[i] for i in sorted_indices[:elitism_k]
    ]  # Keep best k individuals
    new_population = (
        elite + offspring[: len(population) - elitism_k]
    )  # Fill with offspring
    return np.array(new_population)


# Genetic Algorithm main function
def genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.1):
    bounds = [-5.12, 5.12]

    # Step 1: Initialize population
    population = initialize_population(pop_size, bounds)

    for gen in range(generations):
        # Step 2: Evaluate fitness
        fitness = evaluate_fitness(population)

        # Step 3: Selection (Tournament)
        offspring = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)

            # Step 4: Crossover & Mutation
            child1, child2 = simulated_binary_crossover(parent1, parent2)
            child1 = gaussian_mutation(child1, mutation_rate)
            child2 = gaussian_mutation(child2, mutation_rate)

            offspring.extend([child1, child2])

        # Step 5: Replacement (Elitism)
        population = replacement(population, offspring, fitness)

    # Step 6: Return best solution found
    final_fitness = evaluate_fitness(population)
    best_index = np.argmax(final_fitness)
    return population[best_index], final_fitness[best_index]


# Run the Genetic Algorithm
best_solution, best_fitness = genetic_algorithm()
print(f"Best solution: x = {best_solution:.5f}, f(x) = {best_fitness:.5f}")
