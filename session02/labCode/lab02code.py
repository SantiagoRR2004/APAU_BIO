import random
import numpy as np
import tqdm
from abc import ABC, abstractmethod

### `machine_learning_task.py` (Esqueleto de Tarea de ML Especializada)


class MLTask(ABC):
    """
    Esta clase busca los mejores hiperparámetros para una clasificación.
    Esta clase debe contener los datos y definir cómo:
      - crear un individuo
      - calcular el fitness
      - hacer crossover
      - mutar los individuos
    """

    @abstractmethod
    def __init__(self, seed=None):
        """
        :param data: podría ser un conjunto de datos de entrenamiento, o un array de features, etc.
        :param k: parámetro de ejemplo (número de clústeres u otro objetivo).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    @abstractmethod
    def create_individual(self) -> object:
        """
        Retorna una solución (individuo) aleatoria.
        Ejemplos:
          - Para clustering: lista de k*dim floats aleatorios.
          - Para clasificación: conjunto de pesos o hiperparámetros.
          - Para regresión simbólica: estructura de árbol o ecuación linear.
        """
        pass

    @abstractmethod
    def fitness_function(self, individual: object) -> float:
        """
        Evalúa la calidad del individuo y retorna
        un valor numérico (cuanto más alto, mejor).
        Ejemplos:
          - Clustering: -SSE (SSE negativo)
          - Clasificación: exactitud en validación
          - Regresión: -ECM
        """
        pass

    @abstractmethod
    def crossover(
        self, parent1: object, parent2: object, crossover_rate: float
    ) -> tuple:
        """
        Retorna dos 'hijos'. Tal vez no hacer nada si random.random() > crossover_rate.
        """
        pass

    @abstractmethod
    def mutation(self, individual: object, mutation_rate: float) -> object:
        """
        Muta el individuo in-place o crea uno nuevo.
        Ejemplos: desplazamiento aleatorio de parámetros, flip de bits, etc.
        """
        pass

    @abstractmethod
    def generate_table(self) -> object:
        """
        Genera una tabla con los mejores individuos y sus fitnesses.
        """
        pass

    @abstractmethod
    def update_table(
        self, table: object, gen: int, best_ind: dict, best_fit: float
    ) -> None:
        """
        Actualiza la tabla con los mejores individuos y sus fitnesses.
        """
        pass


class GeneticAlgorithm:
    """
    Un marco genérico de Algoritmo Genético.
    """

    def __init__(
        self,
        pop_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        patience=10,
        min_delta=1e-3,
        seed=None,
    ):
        """
        :param pop_size: tamaño de la población
        :param generations: número máximo de generaciones
        :param crossover_rate: probabilidad de realizar crossover
        :param mutation_rate: probabilidad de mutar cada gen
        :param patience: detención temprana si no hay mejora durante 'patience' generaciones
        :param min_delta: mínima mejora para resetear la paciencia
        :param seed: semilla aleatoria opcional
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.min_delta = min_delta

        # Campos opcionales para seguimiento
        self.best_fitness_per_generation = []
        self.global_best = None
        self.global_best_fit = -float("inf")

    def run(self, ml_task: MLTask) -> object:
        """
        Bucle principal del GA:
          - crear población inicial (mediante ml_task.create_individual)
          - evaluar fitness (ml_task.fitness_function)
          - selección, crossover, mutación
          - seguimiento del mejor global
          - detención temprana
        :param ml_task: instancia que proporciona create_individual(), fitness_function(),
                        crossover(), mutation().

        :return: mejor individuo encontrado y lista de valores de fitness por generación.
        """
        # TODO: Implementar los pasos del GA
        # 1) Crear población inicial
        # 2) Evaluar fitness
        # 3) Controlar mejores soluciones
        # 4) for gen in range(self.generations):
        #    - construir nueva población con selección -> crossover -> mutación
        #    - checks de detención temprana
        #    - registrar mejor fitness
        population = []
        for i in range(self.pop_size):
            population.append(ml_task.create_individual())

        best_performers = []

        global_best = None
        global_best_fit = -float("inf")

        no_improvement_counter = 0

        table = ml_task.generate_table()

        # Progress bar
        pbar = tqdm.tqdm(range(self.generations))

        for gen in range(self.generations):
            # Evaluar fitness
            fitnesses = []
            for ind in population:
                f = ml_task.fitness_function(ind)
                fitnesses.append(f)

            # Best of this generation
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx]
            best_fit = fitnesses[best_idx]

            # -- Update "global best" if this generation's best is better --
            if best_fit > global_best_fit + self.min_delta:
                global_best = best_ind
                global_best_fit = best_fit
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.patience:
                # Close progress bar
                pbar.close()
                print(f"Early stopping at generation {gen} due to no improvement.")
                break

            best_performers.append((best_ind, best_fit))

            if table is not None:
                ml_task.update_table(table, gen, best_ind, best_fit)

            # -- Selection --
            selected = self._tournament_selection(population, fitnesses)

            # -- Crossover & Mutation --
            new_pop = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i + 1) % len(selected)]
                c1, c2 = ml_task.crossover(p1, p2, self.crossover_rate)
                c1 = ml_task.mutation(c1, self.mutation_rate)
                c2 = ml_task.mutation(c2, self.mutation_rate)
                new_pop.append(c1)
                new_pop.append(c2)

            population = new_pop[: self.pop_size]

            # Update progress bar
            pbar.update(1)

        # After we finish all generations:
        if table is not None:
            print(table)

        final_fitnesses = [ml_task.fitness_function(ind) for ind in population]
        final_best = population[np.argmax(final_fitnesses)]
        final_best_fit = max(final_fitnesses)
        if final_best_fit > global_best_fit:
            global_best = final_best
            global_best_fit = final_best_fit

        print("\n=== Final Reported Best Solution ===")
        print(f"Global best: {global_best}")
        print(f"Global best fitness: {global_best_fit:.4f}")

        return global_best

    def _tournament_selection(
        self, population: list, fitnesses: list, tsize: int = 3
    ) -> list:
        """
        Ejemplo: selección por torneo para escoger un padre.
        """
        # TODO: implementar o permitir que lo hagan los estudiantes
        selected = []
        zipped = list(zip(population, fitnesses))
        for _ in range(len(population)):
            tournament = random.sample(zipped, tsize)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
