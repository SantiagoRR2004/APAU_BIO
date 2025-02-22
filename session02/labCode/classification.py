import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from lab02code import GeneticAlgorithm, MLTask


class MLTaskClassification(MLTask):
    """
    Esta clase busca los mejores hiperparámetros para una clasificación.
    Esta clase debe contener los datos y definir cómo:
      - crear un individuo
      - calcular el fitness
      - hacer crossover
      - mutar los individuos
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        hyperparameters_bounds: dict = {
            "n_estimators": (10, 200),
            "max_depth": (1, 15),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 5),
            "max_features": (1, 5),
        },
        seed=None,
    ):
        """
        :param data: podría ser un conjunto de datos de entrenamiento, o un array de features, etc.
        :param k: parámetro de ejemplo (número de clústeres u otro objetivo).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.hyperparameters_bounds = hyperparameters_bounds
        # Posiblemente derivar otros parámetros (p.ej., params_per_ind, etc.)

    def create_individual(self) -> dict:
        """
        Retorna una solución (individuo) aleatoria.
        Ejemplos:
          - Para clustering: lista de k*dim floats aleatorios.
          - Para clasificación: conjunto de pesos o hiperparámetros.
          - Para regresión simbólica: estructura de árbol o ecuación linear.
        """
        hyperparameters = {}
        for key in self.hyperparameters_bounds.keys():
            hyperparameters[key] = random.randint(
                self.hyperparameters_bounds[key][0],
                self.hyperparameters_bounds[key][1],
            )

        return hyperparameters

    def fitness_function(self, individual: dict) -> float:
        """
        Evalúa la calidad del individuo y retorna
        un valor numérico (cuanto más alto, mejor).
        Ejemplos:
          - Clustering: -SSE (SSE negativo)
          - Clasificación: exactitud en validación
          - Regresión: -ECM
        """
        random_forest = RandomForestClassifier(**individual)
        random_forest.fit(self.X_train, self.y_train)
        return random_forest.score(self.X_val, self.y_val)

    def crossover(self, parent1: dict, parent2: dict, crossover_rate: float) -> tuple:
        """
        Retorna dos 'hijos'. Tal vez no hacer nada si random.random() > crossover_rate.
        """
        if random.random() > crossover_rate:
            return parent1, parent2
        else:
            alpha = random.random()
            child1 = {}
            child2 = {}
            for key in parent1.keys():
                child1[key] = int(alpha * parent1[key] + (1 - alpha) * parent2[key])
                child2[key] = int(alpha * parent2[key] + (1 - alpha) * parent1[key])

            return child1, child2

    def mutation(self, individual: dict, mutation_rate: float) -> dict:
        """
        Muta el individuo in-place o crea uno nuevo.
        Ejemplos: desplazamiento aleatorio de parámetros, flip de bits, etc.
        """
        if random.random() > mutation_rate:
            return individual
        else:
            # Modificamos un parámetro aleatoriamente
            key = random.choice(list(individual.keys()))
            signo = random.choice([-1, 1])
            lower_bound, upper_bound = self.hyperparameters_bounds[key]
            individual[key] = individual[key] + signo * random.randint(
                lower_bound, upper_bound
            )
            # Evitamos valores fuera de rango
            individual[key] = int(max(min(individual[key], upper_bound), lower_bound))

            return individual

    def score_test(self, individual: dict) -> float:
        """
        Evalua el modelo con los datos de test.
        """
        random_forest = RandomForestClassifier(**individual)
        random_forest.fit(self.X_train, self.y_train)
        return random_forest.score(self.X_test, self.y_test)

    def generate_table(self) -> object:
        """
        Genera una tabla con los mejores individuos y sus fitnesses.
        """
        try:
            from prettytable import PrettyTable

            table = PrettyTable()
            hyperparameters_name = list(self.hyperparameters_bounds.keys())
            table.field_names = ["Gen", "Fitness"] + hyperparameters_name
        except ImportError:
            table = None
        return table

    def update_table(
        self, table: object, gen: int, best_ind: dict, best_fit: float
    ) -> None:
        """
        Actualiza la tabla con los mejores individuos y sus fitnesses.
        """
        hyperparameters = self.hyperparameters_bounds.keys()
        hyperparameters_values = [best_ind[key] for key in hyperparameters]
        table.add_row([gen, round(best_fit, 3)] + hyperparameters_values)


if __name__ == "__main__":

    # Get actual folder
    directory = os.path.dirname(os.path.realpath(__file__))

    # Get dataset path
    datasetPath = os.path.realpath(
        os.path.join(os.path.join(directory, "data"), "schizophrenia_dataset.csv")
    )

    df = pd.read_csv(datasetPath)
    X = df.drop(columns=["MedicationAdherence"])
    y = df["MedicationAdherence"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    ml_task = MLTaskClassification(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=42
    )
    ga = GeneticAlgorithm(seed=42)
    best_sol = ga.run(ml_task=ml_task)
    print(best_sol)

    print(f"Test score: {ml_task.score_test(best_sol)}")
