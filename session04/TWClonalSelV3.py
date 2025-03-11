import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


##############################################################################
# (A) Generate synthetic time-series windows (normal vs. anomalies)
##############################################################################
def generate_single_timeseries_with_anomalies(
    n_points=400,
    anomaly_intervals=[(100, 120), (250, 270)],
    window_size=20,
    step=20,
    random_seed=42,
):
    """
    Create a 1D time series of length n_points, with certain intervals designated as anomalies.
    Then segment into windows of length window_size, stepping by 'step', and label each window
    as anomaly if it overlaps any anomaly interval.
    """
    np.random.seed(random_seed)

    # 1) Build base normal wave
    t_axis = np.linspace(0, 4 * np.pi, n_points)
    base_amp = 1.0
    wave = base_amp * np.sin(t_axis)
    noise = 0.1 * np.random.randn(n_points)
    T = wave + noise

    # 2) Insert anomalies in the specified intervals
    for start_idx, end_idx in anomaly_intervals:
        # triple amplitude + bigger noise in these intervals
        T[start_idx:end_idx] = 3.0 * base_amp * np.sin(t_axis[start_idx:end_idx])
        T[start_idx:end_idx] += 0.3 * np.random.randn(end_idx - start_idx)

    # 3) Slice into windows
    window_starts = range(0, n_points - window_size + 1, step)
    X, y = [], []
    for ws in window_starts:
        we = ws + window_size
        window_data = T[ws:we]
        label = 0  # default = normal
        # If the window overlaps any anomaly interval => label=1
        for a_start, a_end in anomaly_intervals:
            if not (we <= a_start or ws >= a_end):
                label = 1
                break
        X.append(window_data)
        y.append(label)

    X = np.array(X)
    y = np.array(y, dtype=int)
    return T, X, y, list(window_starts)


def plot_timeseries_with_windows(
    T,
    anomaly_intervals,
    window_size,
    window_starts,
    y,
    title="Time Series with Windows",
):
    """
    Plot the full time series in blue, anomaly intervals in red,
    and shade each window in green/orange depending on label.
    """
    n_points = len(T)
    plt.figure(figsize=(12, 4))

    # Plot entire series in blue
    plt.plot(np.arange(n_points), T, color="blue", lw=1)

    # Overwrite anomaly intervals in red
    for a_start, a_end in anomaly_intervals:
        plt.plot(np.arange(a_start, a_end), T[a_start:a_end], color="red", lw=1)

    # Shade each window
    for i, ws in enumerate(window_starts):
        we = ws + window_size
        label = y[i]
        color = "orange" if label == 1 else "green"
        plt.axvspan(ws, we, color=color, alpha=0.1)

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Signal Amplitude")
    plt.xlim(0, n_points)
    plt.legend(
        [
            "Full Series (blue=normal, red=anomaly)",
            "Window shading (green=normal, orange=anomaly)",
        ]
    )
    plt.show()


##############################################################################
# (B) Clonal Selection AIS Class (with time-window vectors)
##############################################################################
class ClonalSelectionAIS:
    """
    A simplified clonal selection algorithm for one-class anomaly detection.
    Each 'antibody' is a center in the same dimension as X, measuring
    'affinity' to normal data. We do a rank-based cloning & mutation,
    extended to time-window vectors.

    The main FIX is that the best (rank=0) center gets the MOST clones,
    and the worst center gets the fewest, to avoid stagnation.
    """

    def __init__(
        self,
        pop_size=30,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.1,
        max_gens=10,
        diversity_rate=0.1,
        random_seed=123,
    ):
        """
        Parameters
        ----------
        pop_size : int
            Number of antibodies in the population.
        clone_factor : float
            Controls how many clones each center produces, based on rank.
        beta : float
            Inverse relationship to affinity => mutation_rate = beta / (affinity).
        mutation_std : float
            Base standard deviation for Gaussian mutation (scaled by mutation_rate).
        max_gens : int
            How many iterations (generations).
        diversity_rate : float
            Fraction of population replaced by random new ones each generation.
        random_seed : int
            Seed for reproducibility.
        """
        self.pop_size = pop_size
        self.clone_factor = clone_factor
        self.beta = beta
        self.mutation_std = mutation_std
        self.max_gens = max_gens
        self.diversity_rate = diversity_rate
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.population_ = None  # shape: (pop_size, n_features)
        self.threshold_ = None  # distance threshold
        self.loss_history_ = []

    def _init_population(self, X):
        """
        Initialize random antibodies within (slightly expanded) data bounds.
        """
        mins = X.min(axis=0) - 0.5
        maxs = X.max(axis=0) + 0.5
        n_features = X.shape[1]
        self.population_ = np.random.uniform(
            mins, maxs, size=(self.pop_size, n_features)
        )

    def _affinity(self, antibody, X_normal):
        """
        affinity = 1 / (1 + mean_distance_to_normal)
        Higher means closer to the normal cluster => better coverage.
        """
        dists = np.linalg.norm(X_normal - antibody, axis=1)
        return 1.0 / (1.0 + dists.mean())

    def _evaluate_pop(self, X_normal):
        """
        Compute affinities for the entire population vector.
        """
        return np.array([self._affinity(ab, X_normal) for ab in self.population_])

    def _clone_and_mutate(self, population, affs):
        """
        - Sort by descending affinity
        - The best center (rank=0) gets the largest clone_count
          => clone_count = int(clone_factor * (n_pop - rank))
        - Mutate them using mutation_std scaled by mutation_rate = beta / affinity
        """
        sorted_idx = np.argsort(affs)[::-1]  # best -> worst
        clones_list = []
        eps = 1e-9
        n_pop = len(population)

        for rank, idx in enumerate(sorted_idx):
            parent_aff = affs[idx]
            parent = population[idx].copy()

            # The best center (rank=0) => largest # clones
            # The worst center (rank=n_pop-1) => fewest # clones
            clone_count = int(self.clone_factor * (n_pop - rank))
            if clone_count < 1:
                clone_count = 1

            mut_rate = self.beta / (parent_aff + eps)
            for _ in range(clone_count):
                clone = parent.copy()
                noise = np.random.normal(
                    0, self.mutation_std * mut_rate, size=clone.shape
                )
                clone += noise
                clones_list.append(clone)

        if len(clones_list) == 0:
            return population
        return np.array(clones_list)

    def fit(self, X_normal, X_val=None, y_val=None, pca_2d=None):
        """
        Train the AIS on normal data only.

        If X_val and y_val are provided, we track 'loss' = fraction of misclassified windows
        on the validation set. Otherwise, we track the fraction of normal training windows
        mis-labeled as anomalies.
        """
        self._init_population(X_normal)

        for gen in range(self.max_gens):
            # 1) Evaluate population
            affs = self._evaluate_pop(X_normal)

            # 2) Clone & mutate
            clones = self._clone_and_mutate(self.population_, affs)
            clone_affs = [self._affinity(c, X_normal) for c in clones]
            clone_affs = np.array(clone_affs)

            # 3) Combine population + clones, keep best pop_size
            combined = np.vstack([self.population_, clones])
            combined_affs = np.concatenate([affs, clone_affs])
            best_idx = np.argsort(combined_affs)[::-1][: self.pop_size]
            self.population_ = combined[best_idx]

            # 4) Recompute affinity, do diversity injection
            final_affs = self._evaluate_pop(X_normal)
            n_new = int(self.pop_size * self.diversity_rate)
            if n_new > 0:
                worst_idx = np.argsort(final_affs)[:n_new]
                mins = X_normal.min(axis=0) - 0.5
                maxs = X_normal.max(axis=0) + 0.5
                new_rand = np.random.uniform(
                    mins, maxs, size=(n_new, X_normal.shape[1])
                )
                self.population_[worst_idx] = new_rand

            # 5) Compute threshold = max distance from each normal sample to its nearest antibody
            self.threshold_ = self._compute_threshold(X_normal)

            # 6) Evaluate 'loss' => fraction misclassified
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                loss = np.mean(y_pred != y_val)
            else:
                # measure fraction of normal mislabeled as anomaly
                y_pred_norm = self.predict(X_normal)
                loss = np.mean(y_pred_norm == 1)
            self.loss_history_.append(loss)

            # 7) Optionally plot iteration
            self._plot_iteration(
                X_val if X_val is not None else X_normal,
                y_val if y_val is not None else np.zeros(len(X_normal)),
                gen,
                loss,
                pca_2d,
            )

    def _compute_threshold(self, X_normal):
        """
        The largest min-distance from X_normal to the AIS population => ensures no normal is above threshold.
        """
        min_dists = []
        for x in X_normal:
            dists = np.linalg.norm(self.population_ - x, axis=1)
            min_dists.append(dists.min())
        return max(min_dists)

    def predict(self, X):
        """
        Label X as anomaly (1) if min distance to the population > threshold,
        otherwise normal (0).
        """
        if self.threshold_ is None:
            raise ValueError("Model not fitted yet. threshold_ is None.")

        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            dists = np.linalg.norm(self.population_ - x, axis=1)
            if dists.min() > self.threshold_:
                labels[i] = 1
        return labels

    def _plot_iteration(self, X_data, y_data, gen, loss, pca_2d=None):
        """
        If pca_2d is provided, project X_data to 2D for visualization,
        color by predicted label, show normal(blue) vs. anomaly(red),
        plus population in green, and outline known anomalies in black.
        """
        plt.ion()
        plt.clf()
        plt.title(f"Clonal AIS - Gen {gen+1}, Loss={loss:.3f}")

        if pca_2d is not None and len(X_data) > 0:
            X2 = pca_2d.transform(X_data)
            y_pred = self.predict(X_data)

            # Normal predicted
            plt.scatter(
                X2[y_pred == 0, 0],
                X2[y_pred == 0, 1],
                c="blue",
                alpha=0.5,
                label="Pred Normal",
            )
            # Anomaly predicted
            plt.scatter(
                X2[y_pred == 1, 0],
                X2[y_pred == 1, 1],
                c="red",
                alpha=0.5,
                label="Pred Anomaly",
            )

            # True anomalies => black outline
            anomalies_idx = np.where(y_data == 1)[0]
            plt.scatter(
                X2[anomalies_idx, 0],
                X2[anomalies_idx, 1],
                facecolors="none",
                edgecolors="black",
                marker="o",
                s=80,
                label="True Anomaly",
            )

            # AIS centers in green (X)
            pop2 = pca_2d.transform(self.population_)
            plt.scatter(
                pop2[:, 0], pop2[:, 1], c="green", marker="X", s=80, label="AIS Centers"
            )

        plt.legend()
        plt.pause(0.5)


##############################################################################
# (C) Demo: Train & Evaluate
##############################################################################
if __name__ == "__main__":

    # -----------------------------------------------------
    # 1) Generate a single time series with anomalies
    # -----------------------------------------------------
    anomaly_intervals = [(100, 120), (250, 270)]
    T, X, y, window_starts = generate_single_timeseries_with_anomalies(
        n_points=400,
        anomaly_intervals=anomaly_intervals,
        window_size=100,
        step=20,
        random_seed=42,
    )

    # -----------------------------------------------------
    # 2) Visualize the time series + window labeling
    # -----------------------------------------------------
    plot_timeseries_with_windows(
        T,
        anomaly_intervals,
        window_size=100,
        window_starts=window_starts,
        y=y,
        title="Single Time Series with Marked Windows & Anomalies",
    )

    # -----------------------------------------------------
    # 3) Split into train (normal only) + test
    # -----------------------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )
    X_train_normal = X_train[y_train == 0]  # keep only normal windows for training

    # For 2D plotting with PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=777)
    pca.fit(X)  # Fit on the entire dataset or just normal

    # -----------------------------------------------------
    # 4) Create Clonal Selection AIS & Train
    # -----------------------------------------------------
    ais = ClonalSelectionAIS(
        pop_size=30,
        clone_factor=5,  # main factor for #clones
        beta=1.0,
        mutation_std=0.1,
        max_gens=10,
        diversity_rate=0.1,
        random_seed=123,
    )

    # Fit on normal data, but track performance on (X_test, y_test)
    ais.fit(X_train_normal, X_val=X_test, y_val=y_test, pca_2d=pca)

    # Turn off interactive plotting if desired
    plt.ioff()
    plt.show()

    # -----------------------------------------------------
    # 5) Plot final loss history
    # -----------------------------------------------------
    plt.figure()
    plt.plot(ais.loss_history_, marker="o")
    plt.title("Clonal AIS: Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Fraction Misclassified on Test)")
    plt.grid(True)
    plt.show()

    # -----------------------------------------------------
    # 6) Final Evaluation on Test
    # -----------------------------------------------------
    y_pred = ais.predict(X_test)
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
