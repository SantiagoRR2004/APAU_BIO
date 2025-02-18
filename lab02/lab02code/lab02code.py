import random
import numpy as np


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

    def create_individual(self):
        """
        Retorna una solución (individuo) aleatoria.
        Ejemplos:
          - Para clustering: lista de k*dim floats aleatorios.
          - Para clasificación: conjunto de pesos o hiperparámetros.
          - Para regresión simbólica: estructura de árbol o ecuación linear.
        """
        pass

    def fitness_function(self, individual):
        """
        Evalúa la calidad del individuo y retorna
        un valor numérico (cuanto más alto, mejor).
        Ejemplos:
          - Clustering: -SSE (SSE negativo)
          - Clasificación: exactitud en validación
          - Regresión: -ECM
        """
        pass

    def crossover(self, parent1, parent2, crossover_rate):
        """
        Retorna dos 'hijos'. Tal vez no hacer nada si random.random() > crossover_rate.
        """
        pass

    def mutation(self, individual, mutation_rate):
        """
        Muta el individuo in-place o crea uno nuevo.
        Ejemplos: desplazamiento aleatorio de parámetros, flip de bits, etc.
        """
        pass

    def run(self, ml_task):
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
        pass

    def _tournament_selection(self, population, fitnesses, tsize=3):
        """
        Ejemplo: selección por torneo para escoger un padre.
        """
        # TODO: implementar o permitir que lo hagan los estudiantes
        pass


### `machine_learning_task.py` (Esqueleto de Tarea de ML Especializada)


class MachineLearningTask:
    """
    Esta clase debe contener los datos y definir cómo:
      - crear un individuo
      - calcular el fitness
      - hacer crossover
      - mutar los individuos
    """

    def __init__(self, data, k=3):
        """
        :param data: podría ser un conjunto de datos de entrenamiento, o un array de features, etc.
        :param k: parámetro de ejemplo (número de clústeres u otro objetivo).
        """
        self.data = data
        self.k = k
        # Posiblemente derivar otros parámetros (p.ej., params_per_ind, etc.)
