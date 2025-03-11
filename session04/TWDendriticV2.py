import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

##############################################################################
# (A) Time-series generator
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
# (B) Dendritic Cell Algorithm (toy version)
##############################################################################
class DendriticCell:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.S_pamp = 0.0
        self.S_danger = 0.0
        self.S_safe = 0.0
        self.mature_type = None
        self.indices = []

    def process(self, idx, pamp, danger, safe):
        if self.mature_type is not None:
            return
        self.indices.append(idx)
        self.S_pamp += pamp
        self.S_danger += danger
        self.S_safe += safe
        total = self.S_pamp + self.S_danger + self.S_safe
        # if we exceed threshold -> mature
        if total >= self.threshold:
            if (self.S_pamp + self.S_danger) > self.S_safe:
                self.mature_type = "mature"  # => label=1
            else:
                self.mature_type = "semi"  # => label=0

    def is_available(self):
        return self.mature_type is None


class DendriticCellAlgorithm:
    """
    We'll feed the data multiple epochs. We define signals from the label or
    from a 'distance-based' guess to produce 'danger' or 'safe'.
    """

    def __init__(self, n_dcs=5, threshold=5.0, epochs=5):
        self.n_dcs = n_dcs
        self.threshold = threshold
        self.epochs = epochs
        self.assignments_ = None
        self.loss_history_ = []

    def _create_pool(self):
        return [DendriticCell(threshold=self.threshold) for _ in range(self.n_dcs)]

    def fit(self, X, y, pca_2d=None):
        """
        We assume a supervised scenario: we can derive PAMP/danger/safe signals from y.
        Typically DCA is unsupervised or uses domain signals, but this is a toy approach.
        """
        n_samples = len(X)
        self.assignments_ = np.zeros(n_samples, dtype=int)
        dcs = self._create_pool()

        for epoch in range(self.epochs):
            idx_order = np.arange(n_samples)
            np.random.shuffle(idx_order)
            dc_idx = 0

            for idx in idx_order:
                pamp, danger, safe = self._compute_signals(X[idx], y[idx])
                tries = 0
                while not dcs[dc_idx].is_available():
                    dc_idx = (dc_idx + 1) % self.n_dcs
                    tries += 1
                    if tries > self.n_dcs:
                        # finalize
                        self._finalize_and_reset(dcs)
                        dc_idx = 0
                dcs[dc_idx].process(idx, pamp, danger, safe)
                dc_idx = (dc_idx + 1) % self.n_dcs

            # end epoch
            self._finalize_and_reset(dcs)
            # measure loss
            loss = np.mean(self.assignments_ != y)
            self.loss_history_.append(loss)
            # plot
            self._plot_epoch(X, y, epoch, loss, pca_2d)
            dcs = self._create_pool()

    def _compute_signals(self, x, label):
        """
        Toy approach: If label=1 => pamp=1, danger=some function. If label=0 => safe=some function.
        In practice, you'd have domain signals for DCA.
        """
        pamp = 1.0 if label == 1 else 0.0
        # "danger" if label=1, let's measure a simple amplitude of x
        danger = 0.0
        if label == 1:
            danger = np.abs(x).mean()  # just a toy measure
        # "safe" if label=0
        safe = 0.0
        if label == 0:
            safe = 2.0 - np.abs(x).mean()  # also a toy measure
            if safe < 0:
                safe = 0
        return pamp, danger, safe

    def _finalize_and_reset(self, dcs):
        """
        Label the data based on the DC maturity, then reset them.
        """
        for dc in dcs:
            if dc.mature_type is not None:
                if dc.mature_type == "mature":  # => label=1
                    for idx in dc.indices:
                        self.assignments_[idx] = 1
                else:  # => label=0
                    for idx in dc.indices:
                        self.assignments_[idx] = 0
        # no partial data carrying over
        for dc in dcs:
            dc.reset()

    def _plot_epoch(self, X, y, epoch, loss, pca_2d=None):
        plt.ion()
        plt.clf()
        plt.title(f"DCA - Epoch {epoch+1}, Loss={loss:.3f}")
        if pca_2d is not None:
            X2 = pca_2d.transform(X)
            y_pred = self.assignments_
            plt.scatter(
                X2[y_pred == 0, 0],
                X2[y_pred == 0, 1],
                c="blue",
                alpha=0.5,
                label="Pred=0",
            )
            plt.scatter(
                X2[y_pred == 1, 0],
                X2[y_pred == 1, 1],
                c="red",
                alpha=0.5,
                label="Pred=1",
            )
            # highlight true anomalies
            anom_idx = np.where(y == 1)[0]
            plt.scatter(
                X2[anom_idx, 0],
                X2[anom_idx, 1],
                facecolors="none",
                edgecolors="black",
                marker="o",
                s=80,
                label="True=1",
            )
        plt.legend()
        plt.pause(0.5)

    def predict(self, X):
        """
        For new data, we can do a naive pass: create DCs, feed them once, see how they mature.
        But let's just do a direct approach using the same signals:
        if sum(danger+pamp) > safe => label=1 else 0
        We'll skip real DC logic for brevity.
        """
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            # guess label to compute signals? That is a catch-22 in real DCA.
            # We'll do a distance measure: if mean(|x|) is large => danger
            mean_abs = np.abs(x).mean()
            pamp = 0.0  # we don't know real label
            danger = mean_abs
            safe = 2.0 - mean_abs
            if safe < 0:
                safe = 0
            # if (pamp+danger)>safe => 1
            if (pamp + danger) > safe:
                labels[i] = 1
        return labels


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

    # We'll do a direct train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    # DCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=777).fit(X)
    dca = DendriticCellAlgorithm(n_dcs=5, threshold=5.0, epochs=5)
    dca.fit(X_train, y_train, pca_2d=pca)

    plt.ioff()
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(dca.loss_history_, marker="o")
    plt.title("DCA: Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassification Rate (Train)")
    plt.grid(True)
    plt.show()

    # Final evaluation on test
    y_pred = dca.predict(X_test)
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
