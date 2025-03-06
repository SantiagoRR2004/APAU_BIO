import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

##############################################################################
# (A) Generate synthetic time-series windows (same function as before)
##############################################################################


def generate_single_timeseries_with_anomalies(
    n_points=400,
    anomaly_intervals=[(100, 120), (250, 270)],
    window_size=20,
    step=20,
    random_seed=42,
):
    np.random.seed(random_seed)

    # 1) Build base normal wave
    t_axis = np.linspace(0, 4 * np.pi, n_points)
    base_amp = 1.0
    wave = base_amp * np.sin(t_axis)
    noise = 0.1 * np.random.randn(n_points)
    T = wave + noise

    # 2) Insert anomalies
    for start_idx, end_idx in anomaly_intervals:
        # triple amplitude + bigger noise
        T[start_idx:end_idx] = 3.0 * base_amp * np.sin(t_axis[start_idx:end_idx])
        T[start_idx:end_idx] += 0.3 * np.random.randn(end_idx - start_idx)

    # 3) Slice into windows
    window_starts = range(0, n_points - window_size + 1, step)
    X, y = [], []
    for ws in window_starts:
        we = ws + window_size
        window_data = T[ws:we]
        # label=1 if overlaps any anomaly interval
        label = 0
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
    n_points = len(T)
    plt.figure(figsize=(12, 4))

    # Plot entire series in blue
    plt.plot(np.arange(n_points), T, color="blue", lw=1)

    # Overwrite anomaly intervals in red
    for a_start, a_end in anomaly_intervals:
        plt.plot(np.arange(a_start, a_end), T[a_start:a_end], color="red", lw=1)

    # Draw vertical spans for each window
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
# (B) Negative Selection Class
##############################################################################
class NegativeSelectionVectors:
    """
    Simple negative selection for time-series windows (1D array of length window_size).
    Each detector is a center vector + we have a 'radius'.
    We remove any detector that 'matches' (is within radius of) normal data.
    The rest detect anomalies.
    """

    def __init__(
        self,
        n_detectors=30,
        detector_radius=0.5,
        n_generations=10,
        n_new=10,
        random_seed=123,
    ):
        self.n_detectors = n_detectors
        self.detector_radius = detector_radius
        self.n_generations = n_generations
        self.n_new = n_new
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.detectors_ = None
        self.loss_history_ = []

    def _init_detectors(self, X_normal):
        # bounding box
        mins = X_normal.min(axis=0) - 0.5
        maxs = X_normal.max(axis=0) + 0.5
        D = X_normal.shape[1]

        # generate random detectors
        new_detectors = np.random.uniform(mins, maxs, size=(2 * self.n_detectors, D))
        # filter out self-reactive
        valid = self._filter_self_reactive(new_detectors, X_normal)
        self.detectors_ = valid[: self.n_detectors]

    def _filter_self_reactive(self, detectors, X_normal):
        """
        Remove detectors that match any normal sample (distance < radius).
        """
        filtered = []
        for d in detectors:
            dists = np.linalg.norm(X_normal - d, axis=1)
            if np.min(dists) > self.detector_radius:
                filtered.append(d)
        return np.array(filtered)

    def fit(self, X_normal, X_val=None, y_val=None, pca_2d=None):
        """
        Train negative selection.
        After each generation, we measure loss on X_val,y_val if provided (or on X_normal).
        Then plot in 2D if pca_2d is given.
        """
        self._init_detectors(X_normal)

        for gen in range(self.n_generations):
            # introduce new
            mins = X_normal.min(axis=0) - 0.5
            maxs = X_normal.max(axis=0) + 0.5
            D = X_normal.shape[1]
            new_rand = np.random.uniform(mins, maxs, size=(self.n_new, D))
            valid_new = self._filter_self_reactive(new_rand, X_normal)

            combined = (
                np.vstack([self.detectors_, valid_new])
                if len(valid_new) > 0
                else self.detectors_
            )
            np.random.shuffle(combined)
            combined = combined[: self.n_detectors]
            self.detectors_ = combined

            # Evaluate loss
            loss = None
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                loss = np.mean(y_pred != y_val)
            else:
                # fraction of normal points flagged
                y_pred_norm = self.predict(X_normal)
                loss = np.mean(y_pred_norm == 1)
            self.loss_history_.append(loss)

            # Plot iteration
            self._plot_generation(
                X_val if X_val is not None else X_normal,
                y_val if y_val is not None else np.zeros(len(X_normal)),
                gen,
                loss,
                pca_2d,
            )

    def predict(self, X):
        """
        Label 1 if matched by ANY detector => anomaly
        else 0 => normal
        """
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            dists = np.linalg.norm(self.detectors_ - x, axis=1)
            if np.any(dists < self.detector_radius):
                labels[i] = 1
        return labels

    def _plot_generation(self, X_data, y_data, gen, loss, pca_2d=None):
        """
        If pca_2d is given, project X_data to 2D, color by predicted label,
        show detectors as circle patches or green points, etc.
        """
        plt.ion()
        plt.clf()
        plt.title(f"Negative Selection - Gen {gen+1}, Loss={loss:.3f}")

        if pca_2d is not None and len(X_data) > 0:
            X2 = pca_2d.transform(X_data)
            y_pred = self.predict(X_data)
            # normal pred
            plt.scatter(
                X2[y_pred == 0, 0],
                X2[y_pred == 0, 1],
                c="blue",
                alpha=0.5,
                label="Pred Normal",
            )
            # anomaly pred
            plt.scatter(
                X2[y_pred == 1, 0],
                X2[y_pred == 1, 1],
                c="red",
                alpha=0.5,
                label="Pred Anomaly",
            )

            # show true anomaly as black circles
            anom_idx = np.where(y_data == 1)[0]
            plt.scatter(
                X2[anom_idx, 0],
                X2[anom_idx, 1],
                facecolors="none",
                edgecolors="black",
                marker="o",
                s=80,
                label="True Anomaly",
            )

            # plot detectors
            det2 = pca_2d.transform(self.detectors_)
            plt.scatter(
                det2[:, 0], det2[:, 1], c="green", marker="X", s=60, label="Detectors"
            )

        plt.legend()
        plt.pause(0.5)


##############################################################################
# (C) Demo
##############################################################################
if __name__ == "__main__":
    # 1) Generate
    anomaly_intervals = [(100, 120), (250, 270)]
    T, X, y, window_starts = generate_single_timeseries_with_anomalies(
        n_points=400,
        anomaly_intervals=anomaly_intervals,
        window_size=100,
        step=20,
        random_seed=42,
    )
    # 2) Plot
    plot_timeseries_with_windows(
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

    # PCA for 2D plotting
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=777).fit(X)

    nsa = NegativeSelectionVectors(
        n_detectors=30, detector_radius=0.8, n_generations=10, n_new=10, random_seed=123
    )
    nsa.fit(X_train_normal, X_val=X_test, y_val=y_test, pca_2d=pca)

    plt.ioff()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(nsa.loss_history_, marker="o")
    plt.title("Negative Selection: Loss History")
    plt.xlabel("Generation")
    plt.ylabel("Loss (Test Misclassification)")
    plt.grid(True)
    plt.show()

    # Final evaluation
    y_pred = nsa.predict(X_test)
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
