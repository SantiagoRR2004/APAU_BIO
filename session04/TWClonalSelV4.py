import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


##############################################################################
# (A) Generate synthetic time-series windows (normal vs. anomalies)
##############################################################################
def generate_single_timeseries_with_anomalies(
    n_points=400,
    anomaly_intervals=[(100, 120), (250, 270)],
    window_size=20,
    step=20,
    random_seed=42
):
    """
    Create a 1D time series of length n_points, with certain intervals designated as anomalies.
    Then segment into windows of length window_size, stepping by 'step', and label each window
    as anomaly if it overlaps any anomaly interval.
    """
    np.random.seed(random_seed)

    # 1) Build base normal wave
    t_axis = np.linspace(0, 4*np.pi, n_points)
    base_amp = 1.0
    wave = base_amp * np.sin(t_axis)
    noise = 0.1 * np.random.randn(n_points)
    T = wave + noise

    # 2) Insert anomalies in the specified intervals
    for (start_idx, end_idx) in anomaly_intervals:
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
        for (a_start, a_end) in anomaly_intervals:
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
    title="Time Series with Windows"
):
    """
    Plot the full time series in blue, anomaly intervals in red,
    and shade each window in green/orange depending on label.
    """
    n_points = len(T)
    plt.figure(figsize=(12, 4))

    # Plot entire series in blue
    plt.plot(np.arange(n_points), T, color='blue', lw=1)

    # Overwrite anomaly intervals in red
    for (a_start, a_end) in anomaly_intervals:
        plt.plot(np.arange(a_start, a_end), T[a_start:a_end], color='red', lw=1)

    # Shade each window
    for i, ws in enumerate(window_starts):
        we = ws + window_size
        label = y[i]
        color = 'orange' if label == 1 else 'green'
        plt.axvspan(ws, we, color=color, alpha=0.1)

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Signal Amplitude")
    plt.xlim(0, n_points)
    plt.legend([
        "Full Series (blue=normal, red=anomaly)",
        "Window shading (green=normal, orange=anomaly)"
    ])
    plt.show()


##############################################################################
# (B) Clonal Selection AIS Class (Time-Series Windows)
##############################################################################
class ClonalSelectionAIS:
    """
    A simplified clonal selection algorithm for one-class anomaly detection
    on time-window vectors. We track coverage-based loss each iteration
    rather than classification-based, so we can see improvement.

    Key fix:
    - The best (rank=0) center gets the MOST clones, worst gets fewer.
    - We measure coverage-based loss => sum of min distances from normal data
      to the AIS population. This should decrease if the population is better
      covering the normal class.
    - At the end, we set a threshold for anomaly detection. By default, the
      threshold is the MAX distance from normal data, but you can set a
      percentile if you wish.
    """

    def __init__(self,
                 pop_size=30,
                 clone_factor=5,
                 beta=1.0,
                 mutation_std=0.1,
                 max_gens=10,
                 diversity_rate=0.1,
                 threshold_percentile=1.0,  # 1.0 => max distance, e.g., 0.95 => 95th percentile
                 random_seed=123):
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
        threshold_percentile : float
            1.0 => use max distance from normal data as threshold,
            0.95 => use 95th percentile distance, etc.
        random_seed : int
            Seed for reproducibility.
        """
        self.pop_size = pop_size
        self.clone_factor = clone_factor
        self.beta = beta
        self.mutation_std = mutation_std
        self.max_gens = max_gens
        self.diversity_rate = diversity_rate
        self.threshold_percentile = threshold_percentile
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.population_ = None   # shape: (pop_size, n_features)
        self.threshold_ = None    # distance threshold
        self.coverage_loss_history_ = []  # sum of min distances (training coverage)
        self.n_features_ = None

    ############### Core AIS Steps ###############
    def _init_population(self, X):
        """
        Initialize random antibodies within slightly expanded data bounds.
        """
        mins = X.min(axis=0) - 0.5
        maxs = X.max(axis=0) + 0.5
        self.n_features_ = X.shape[1]
        self.population_ = np.random.uniform(mins, maxs,
                                             size=(self.pop_size, self.n_features_))

    def _affinity(self, center, X_normal):
        """
        affinity = 1 / (1 + mean_distance_to_normal)
        """
        dists = np.linalg.norm(X_normal - center, axis=1)
        avg_dist = np.mean(dists)
        return 1.0 / (1.0 + avg_dist)

    def _evaluate_pop_affinity(self, X_normal):
        """
        Returns array of shape (pop_size,) with each center's affinity.
        """
        return np.array([self._affinity(ab, X_normal) for ab in self.population_])

    def _coverage_loss(self, X_normal):
        """
        Sum of min distances from each normal sample to the nearest center.
        Lower => better coverage.
        """
        dist_matrix = cdist(X_normal, self.population_)
        # For each row (each normal), get min distance
        min_dists = dist_matrix.min(axis=1)
        return min_dists.sum()

    def _clone_and_mutate(self, population, affinities):
        """
        - Sort by descending affinity
        - The best center gets the most clones => clone_count = clone_factor*(pop_size-rank)
        - Mutate with Gaussian noise (std scaled by (beta / affinity)).
        """
        sorted_idx = np.argsort(affinities)[::-1]  # best -> worst
        clones_list = []
        eps = 1e-9
        n_pop = len(population)

        for rank, idx in enumerate(sorted_idx):
            parent_aff = affinities[idx]
            parent = population[idx].copy()
            # Best rank=0 => largest # clones
            clone_count = int(self.clone_factor * (n_pop - rank))
            if clone_count < 1:
                clone_count = 1

            mutation_rate = self.beta / (parent_aff + eps)

            for _ in range(clone_count):
                clone = parent.copy()
                noise = np.random.normal(
                    0, self.mutation_std * mutation_rate, size=clone.shape
                )
                clone += noise
                clones_list.append(clone)

        return np.array(clones_list) if len(clones_list) > 0 else population

    def fit(self, X_normal):
        """
        Train the AIS purely on normal data. Each generation, we measure
        coverage-based loss (sum of min distances) to see if coverage improves.

        Finally, threshold_ is set to either the max or a percentile-based distance
        from normal data. This is used in predict() to label anomalies.
        """
        # 1) Initialize population if needed
        self._init_population(X_normal)

        for gen in range(self.max_gens):
            # 2) Evaluate affinity
            affs = self._evaluate_pop_affinity(X_normal)

            # 3) Clone & mutate
            clones = self._clone_and_mutate(self.population_, affs)

            # 4) Evaluate clones => keep top pop_size
            clone_affs = np.array([self._affinity(c, X_normal) for c in clones])
            combined_pop = np.vstack([self.population_, clones])
            combined_affs = np.concatenate([affs, clone_affs])
            best_idx = np.argsort(combined_affs)[::-1][:self.pop_size]
            self.population_ = combined_pop[best_idx]

            # 5) Diversity injection
            final_affs = self._evaluate_pop_affinity(X_normal)
            n_new = int(self.pop_size * self.diversity_rate)
            if n_new > 0:
                worst_idx = np.argsort(final_affs)[:n_new]
                mins = X_normal.min(axis=0) - 0.5
                maxs = X_normal.max(axis=0) + 0.5
                new_rand = np.random.uniform(mins, maxs, size=(n_new, self.n_features_))
                self.population_[worst_idx] = new_rand

            # 6) Compute coverage loss & store
            c_loss = self._coverage_loss(X_normal)
            self.coverage_loss_history_.append(c_loss)

        # 7) After final iteration, set threshold
        self._set_threshold(X_normal)

    def _set_threshold(self, X_normal):
        """
        Sets threshold based on a percentile of min distances from X_normal to the population.
        If threshold_percentile=1.0, we do max distance => ensures all normal are labeled normal.
        If threshold_percentile < 1.0, we do a partial approach, e.g. 0.95 => 95th percentile.
        """
        dist_mat = cdist(X_normal, self.population_)
        min_dists = dist_mat.min(axis=1)
        if self.threshold_percentile >= 1.0:
            self.threshold_ = np.max(min_dists)
        else:
            pct_val = np.percentile(min_dists, self.threshold_percentile * 100)
            self.threshold_ = pct_val

    ############### Predict ###############
    def predict(self, X):
        """
        Label X as anomaly (1) if min distance to the population > threshold,
        otherwise normal (0).
        """
        if self.threshold_ is None:
            raise ValueError("Model not fitted: threshold is None.")

        dist_mat = cdist(X, self.population_)
        min_dists = dist_mat.min(axis=1)
        return (min_dists > self.threshold_).astype(int)


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
        random_seed=42
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
        title="Single Time Series with Marked Windows & Anomalies"
    )

    # -----------------------------------------------------
    # 3) Split into train (normal only) + test
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )
    X_train_normal = X_train[y_train == 0]  # keep only normal windows for training

    # For optional 2D plotting, we'll do PCA if desired
    # (But here we'll just watch coverage-based loss)
    pca = PCA(n_components=2, random_state=777)
    pca.fit(X)

    # -----------------------------------------------------
    # 4) Create & Train Clonal AIS
    # -----------------------------------------------------
    ais = ClonalSelectionAIS(
        pop_size=30,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.1,
        max_gens=10,
        diversity_rate=0.1,
        threshold_percentile=1.0,  # use max distance from normal
        random_seed=123
    )
    ais.fit(X_train_normal)

    # -----------------------------------------------------
    # 5) Coverage Loss History
    # -----------------------------------------------------
    plt.figure()
    plt.plot(ais.coverage_loss_history_, marker='o')
    plt.title("AIS Coverage Loss (Sum of Min Distances)")
    plt.xlabel("Generation")
    plt.ylabel("Coverage Loss")
    plt.grid(True)
    plt.show()

    # -----------------------------------------------------
    # 6) Evaluate on test set
    # -----------------------------------------------------
    y_pred = ais.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Test):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal","Anomaly"]))

    # If you want to see how many test anomalies were caught vs. missed, check the confusion matrix.
