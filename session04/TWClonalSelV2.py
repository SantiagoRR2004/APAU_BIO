import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import timeseries


##############################################################################
# (B) Clonal Selection AIS Class
##############################################################################
class ClonalSelectionAIS:
    """
    A simplified clonal selection algorithm for one-class anomaly detection.
    Each 'antibody' is a center in the same dimension as X, plus we measure
    an 'affinity' to normal data. We do a rank-based cloning & mutation,
    similar to the 2D examples but extended to time-window vectors.
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
        Initialize random antibodies within data bounds.
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
        return np.array([self._affinity(ab, X_normal) for ab in self.population_])

    def _clone_and_mutate(self, population, affs):
        """
        - Sort by descending affinity
        - For each antibody, produce clones proportional to rank
        - Mutate them
        """
        sorted_idx = np.argsort(affs)[::-1]
        clones_list = []
        eps = 1e-9
        for rank, idx in enumerate(sorted_idx):
            parent_aff = affs[idx]
            parent = population[idx].copy()
            clone_count = int(self.clone_factor * (rank + 1))
            for _ in range(clone_count):
                clone = parent.copy()
                mut_rate = self.beta / (parent_aff + eps)
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
        Train the AIS on normal data only. If X_val, y_val provided,
        we can measure a 'loss' = fraction of normal points labeled anomaly.
        (Optionally incorporate anomalies in y_val to measure total error.)
        pca_2d: if provided, used for plotting in 2D.
        """
        self._init_population(X_normal)

        for gen in range(self.max_gens):
            # Evaluate
            affs = self._evaluate_pop(X_normal)

            # Clone & mutate
            clones = self._clone_and_mutate(self.population_, affs)
            clone_affs = [self._affinity(c, X_normal) for c in clones]
            clone_affs = np.array(clone_affs)

            # Combine
            combined = np.vstack([self.population_, clones])
            combined_affs = np.concatenate([affs, clone_affs])

            # Keep best pop_size
            best_idx = np.argsort(combined_affs)[::-1][: self.pop_size]
            self.population_ = combined[best_idx]

            # ---- Fix starts here ----
            # Recompute affinity for just the final population
            final_affs = self._evaluate_pop(X_normal)

            # Diversity injection
            n_new = int(self.pop_size * self.diversity_rate)
            if n_new > 0:
                worst_idx = np.argsort(final_affs)[:n_new]
                # create new random
                mins = X_normal.min(axis=0) - 0.5
                maxs = X_normal.max(axis=0) + 0.5
                new_rand = np.random.uniform(
                    mins, maxs, size=(n_new, X_normal.shape[1])
                )
                self.population_[worst_idx] = new_rand

            # Compute threshold = max distance from normal to nearest antibody
            # so all training normal => inliers
            self.threshold_ = self._compute_threshold(X_normal)

            # Evaluate loss on val set if provided
            loss = None
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                # simple misclassification
                loss = np.mean(y_pred != y_val)
                self.loss_history_.append(loss)
            else:
                # We'll measure fraction of normal points mis-labeled as anomaly
                y_pred_norm = self.predict(X_normal)
                loss = np.mean(y_pred_norm == 1)
                self.loss_history_.append(loss)

            # Plot iteration
            self._plot_iteration(
                X_val if X_val is not None else X_normal,
                y_val if y_val is not None else np.zeros(len(X_normal)),
                gen,
                loss,
                pca_2d,
            )

    def _compute_threshold(self, X_normal):
        """
        max distance from each normal sample to its nearest antibody
        => ensures no training normal is outside threshold.
        """
        min_dists = []
        for x in X_normal:
            dists = np.linalg.norm(self.population_ - x, axis=1)
            min_dists.append(dists.min())
        return max(min_dists)

    def predict(self, X):
        """
        Label X as anomaly (1) if min distance to the population > threshold,
        else normal (0).
        """
        if self.threshold_ is None:
            raise ValueError("Model not fitted, threshold is None.")
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
        plus population in green.
        """
        plt.ion()
        plt.clf()
        title = f"Clonal Selection AIS - Gen {gen+1}, Loss={loss:.3f}"
        plt.title(title)

        if pca_2d is not None and len(X_data) > 0:
            X2 = pca_2d.transform(X_data)
            y_pred = self.predict(X_data)
            # Normal
            plt.scatter(
                X2[y_pred == 0, 0],
                X2[y_pred == 0, 1],
                c="blue",
                alpha=0.5,
                label="Pred Normal",
            )
            # Anomaly
            plt.scatter(
                X2[y_pred == 1, 0],
                X2[y_pred == 1, 1],
                c="red",
                alpha=0.5,
                label="Pred Anomaly",
            )
            # True label outlines
            # (We'll just outline anomalies in black)
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

            # Plot population
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

    # 1) Generate
    anomaly_intervals = [(100, 120), (250, 270)]
    T, X, y, window_starts = timeseries.generate_single_timeseries_with_anomalies(
        n_points=400,
        anomaly_intervals=anomaly_intervals,
        window_size=100,
        step=20,
        random_seed=42,
    )
    # 2) Plot
    timeseries.plot_timeseries_with_windows(
        T,
        anomaly_intervals,
        window_size=100,
        window_starts=window_starts,
        y=y,
        title="Single Time Series with Marked Windows & Anomalies",
    )

    # 2) Split into train (only normal) + test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )
    X_train_normal = X_train[y_train == 0]

    # For 2D plotting, we'll use PCA on the entire dataset
    pca = PCA(n_components=2, random_state=777)
    pca.fit(X)  # or fit on X_train_normal

    # 3) Create Clonal Selection AIS
    ais = ClonalSelectionAIS(
        pop_size=30,
        clone_factor=5,
        beta=1.0,
        mutation_std=0.1,
        max_gens=10,
        diversity_rate=0.1,
        random_seed=123,
    )

    # 4) Fit on normal data
    ais.fit(X_train_normal, X_val=X_test, y_val=y_test, pca_2d=pca)

    # Turn off interactive & show final
    plt.ioff()
    plt.show()

    # 5) Plot final loss history
    plt.figure()
    plt.plot(ais.loss_history_, marker="o")
    plt.title("Clonal AIS: Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Test Misclassification)")
    plt.grid(True)
    plt.show()

    # 6) Final Evaluation
    y_pred = ais.predict(X_test)
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
