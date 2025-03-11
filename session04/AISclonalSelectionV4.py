import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cdist

class ClonalAISAnomaly:
    """
    Clonal Selection AIS for One-Class Anomaly Detection with Interactive Plotting
    -----------------------------------------------------------------------------
    - Each antibody is a center in feature space.
    - Affinity = 1 / (1 + avg_distance_to_normal_samples).
    - 'Loss' = sum of (distance from each normal sample to its nearest center).
      This is used to monitor coverage during training.
    """

    def __init__(
        self,
        pop_size=30,
        clone_factor=5,
        mutation_std=0.1,
        beta=1.0,
        n_generations=20,
        diversity_rate=0.1,
        visualize=True,
        random_seed=None
    ):
        """
        Parameters
        ----------
        pop_size : int
            Number of antibody centers in the population.
        clone_factor : float
            Factor that determines how many clones each center produces, based on its rank.
        mutation_std : float
            Base standard deviation for Gaussian mutation (scaled by mutation_rate).
        beta : float
            Controls how strongly affinity reduces the mutation rate (mutation_rate = beta / affinity).
        n_generations : int
            How many clonal selection iterations to run.
        diversity_rate : float
            Fraction of the population replaced by new random centers each generation.
        visualize : bool
            If True, plots the population each generation (interactive).
        random_seed : int, optional
            Seed for reproducibility.
        """
        self.pop_size = pop_size
        self.clone_factor = clone_factor
        self.mutation_std = mutation_std
        self.beta = beta
        self.n_generations = n_generations
        self.diversity_rate = diversity_rate

        self.visualize = visualize
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Internal variables
        self.population_ = None    # shape: (pop_size, n_features)
        self.n_features_ = None
        self.threshold_ = None     # distance threshold for anomaly detection
        self.loss_history_ = []    # track sum of min distances across normal data each generation

        if self.visualize:
            plt.ion()  # enable interactive mode for real-time plotting

    def _initialize_population(self, X, low=None, high=None):
        """
        Initialize the population with random centers within data bounds (or user-provided).
        """
        if low is None:
            low = X.min(axis=0)
        if high is None:
            high = X.max(axis=0)

        self.n_features_ = X.shape[1]
        self.population_ = np.random.uniform(low, high, size=(self.pop_size, self.n_features_))

    def _compute_affinities(self, X_normal, candidates=None):
        """
        Vectorized affinity computation:
          affinity = 1 / (1 + average_distance_to_normal_samples)

        Parameters
        ----------
        X_normal : ndarray of shape (n_normal, n_features)
        candidates : ndarray of shape (n_candidates, n_features) or None
            If None, use self.population_.

        Returns
        -------
        affs : ndarray of shape (n_candidates,)
        """
        if candidates is None:
            candidates = self.population_

        # distances.shape => (n_candidates, n_normal)
        distances = cdist(candidates, X_normal)
        avg_dists = distances.mean(axis=1)
        affs = 1.0 / (1.0 + avg_dists)
        return affs

    def _compute_loss(self, X_normal):
        """
        'Loss' = sum of min distances from each normal sample to the nearest center.
        Lower is better coverage of normal data.
        """
        # distances.shape => (n_normal, pop_size)
        distances = cdist(X_normal, self.population_)
        min_dists = distances.min(axis=1)
        return min_dists.sum()

    def _clone_and_mutate(self, population, affinities):
        """
        1) Sort centers by descending affinity (best -> worst).
        2) For each center in that order, produce clones. The best center gets the most clones.
        3) Mutate clones using Gaussian noise (std = mutation_std * mutation_rate).
           mutation_rate = beta / (affinity + eps).
        4) Return all clones as a single array.
        """
        eps = 1e-9
        sorted_indices = np.argsort(affinities)[::-1]  # best to worst

        clones_list = []
        n_pop = len(population)

        for rank, idx in enumerate(sorted_indices):
            parent_aff = affinities[idx]
            parent = population[idx]

            # The best center (rank=0) gets the largest # of clones => (pop_size - rank)
            # The worst center (rank=pop_size-1) gets the smallest # => 1
            clone_count = int(self.clone_factor * (n_pop - rank))
            if clone_count < 1:
                clone_count = 1

            mutation_rate = self.beta / (parent_aff + eps)
            for _ in range(clone_count):
                clone = parent.copy()
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.mutation_std * mutation_rate,
                    size=self.n_features_
                )
                clone += noise
                clones_list.append(clone)

        if len(clones_list) == 0:
            return np.empty((0, self.n_features_))
        return np.array(clones_list)

    def _update_population(self, X_normal, clones, keep_ratio=0.2):
        """
        1) Compute affinities for all clones.
        2) Keep top fraction of clones (by affinity).
        3) Merge them with the original population.
        4) Evaluate the merged population's affinities.
        5) Keep the top pop_size overall.
        """
        if clones.shape[0] == 0:
            return self.population_

        # Evaluate clone affinities
        clone_affs = self._compute_affinities(X_normal, clones)

        # Keep top fraction of clones
        n_keep = max(1, int(keep_ratio * len(clones)))
        top_clone_indices = np.argsort(clone_affs)[::-1][:n_keep]
        best_clones = clones[top_clone_indices]

        # Merge them with the existing population
        merged_pop = np.vstack([self.population_, best_clones])
        merged_affs = self._compute_affinities(X_normal, merged_pop)

        # Retain the best pop_size from the merged set
        top_indices = np.argsort(merged_affs)[::-1][:self.pop_size]
        new_pop = merged_pop[top_indices]
        return new_pop

    def fit(self, X_normal):
        """
        Train the AIS on normal data only.
        """
        # 1) Initialize population if needed
        if self.population_ is None:
            self._initialize_population(X_normal)

        for gen in range(self.n_generations):
            # 2) Evaluate current affinities
            affs = self._compute_affinities(X_normal, self.population_)

            # 3) Clone & mutate
            clones = self._clone_and_mutate(self.population_, affs)

            # 4) Update population
            self.population_ = self._update_population(X_normal, clones, keep_ratio=0.2)

            # 5) Diversity injection
            n_new = int(self.pop_size * self.diversity_rate)
            if n_new > 0:
                pop_affs = self._compute_affinities(X_normal, self.population_)
                worst_indices = np.argsort(pop_affs)[:n_new]  # worst = ascending
                low = X_normal.min(axis=0)
                high = X_normal.max(axis=0)
                new_random = np.random.uniform(low, high, size=(n_new, self.n_features_))
                self.population_[worst_indices] = new_random

            # 6) Compute & store the loss
            loss_val = self._compute_loss(X_normal)
            self.loss_history_.append(loss_val)

            # 7) Visualization (optional)
            if self.visualize:
                self._plot_current_state(X_normal, gen, loss_val)

        # End of training: finalize threshold
        if self.visualize:
            plt.ioff()

        self._set_threshold(X_normal)
        print(f"Final derived threshold for anomalies: {self.threshold_:.4f}")

    def _set_threshold(self, X_normal):
        """
        Sets the threshold = max distance from any normal sample to its nearest center,
        ensuring all training data is labeled 'normal'.
        """
        distances = cdist(X_normal, self.population_)
        min_dists = distances.min(axis=1)
        self.threshold_ = np.max(min_dists)

    def _plot_current_state(self, X_normal, gen, loss_val):
        """
        Interactive plotting of population vs. normal data each generation.
        """
        plt.clf()
        plt.scatter(X_normal[:, 0], X_normal[:, 1], c='blue', alpha=0.5, label='Normal Data')
        plt.scatter(self.population_[:, 0], self.population_[:, 1],
                    facecolors='none', edgecolors='red', s=80, label='AIS Centers')
        plt.title(f"Generation {gen+1}/{self.n_generations}\nLoss = {loss_val:.2f}")
        plt.legend()
        plt.draw()
        plt.pause(0.5)

    def predict(self, X):
        """
        Return binary labels: 0 = normal, 1 = anomaly
        Based on distance to population vs. threshold_.
        """
        if self.threshold_ is None:
            raise ValueError("Cannot predict without a threshold. Call fit() first.")

        distances = cdist(X, self.population_)
        min_dists = distances.min(axis=1)
        return (min_dists > self.threshold_).astype(int)

    def decision_function(self, X):
        """
        Returns the min distance to any AIS center (lower => more normal).
        """
        distances = cdist(X, self.population_)
        min_dists = distances.min(axis=1)
        return min_dists


if __name__ == "__main__":
    # ----------------------------------------------------
    # 1. Generate Synthetic Dataset
    # ----------------------------------------------------
    np.random.seed(42)

    # Two normal clusters
    X_normal_cluster1 = np.random.normal(loc=(-2.0, 0.0), scale=1.0, size=(100, 2))
    X_normal_cluster2 = np.random.normal(loc=( 2.0, 0.0), scale=1.0, size=(100, 2))
    X_normal = np.vstack((X_normal_cluster1, X_normal_cluster2))

    # Anomalies
    # (A) Near overlap anomalies
    X_anomalies_near = np.random.normal(loc=(0.0, 3.0), scale=1.2, size=(40, 2))
    # (B) Uniformly spread anomalies
    X_anomalies_far = np.random.uniform(low=-6.0, high=6.0, size=(20, 2))
    X_anomalies = np.vstack((X_anomalies_near, X_anomalies_far))

    # Labels for anomalies
    y_anomalies = np.ones(60, dtype=int)  # label=1 => anomaly

    # Combine to form a test set
    X_test = np.vstack((X_normal, X_anomalies))
    y_normal = np.zeros(len(X_normal), dtype=int)  # label=0 => normal
    y_test = np.concatenate((y_normal, y_anomalies))

    # ----------------------------------------------------
    # 2. Initialize & Train AIS on normal data
    # ----------------------------------------------------
    ais = ClonalAISAnomaly(
        pop_size=30,
        clone_factor=5,
        mutation_std=0.1,
        beta=1.0,
        n_generations=25,
        diversity_rate=0.15,
        visualize=True,
        random_seed=123
    )
    ais.fit(X_normal)  # train only on normal data

    # Turn off interactive plotting if desired
    plt.ioff()
    plt.show()

    # ----------------------------------------------------
    # 3. Examine the Loss History
    # ----------------------------------------------------
    plt.figure()
    plt.plot(ais.loss_history_, marker='o')
    plt.title("AIS - Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Sum of Min Distances (Loss)")
    plt.grid()
    plt.show()

    # ----------------------------------------------------
    # 4. Prediction & Evaluation
    # ----------------------------------------------------
    y_pred = ais.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    # ----------------------------------------------------
    # 5. Final Plot of AIS vs. Data
    # ----------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(X_normal[:, 0], X_normal[:, 1], c="blue", alpha=0.5, label="Normal")
    plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c="red", alpha=0.7, marker='x', label="Anomalies")
    plt.scatter(
        ais.population_[:, 0], ais.population_[:, 1],
        facecolors='none', edgecolors='magenta', s=120, label="AIS Centers"
    )
    plt.title("Final AIS Centers vs. Data")
    plt.legend()
    plt.show()
