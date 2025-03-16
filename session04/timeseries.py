import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def from_data_to_timeseries(
    data: pd.DataFrame,
    window_size: int = 100,
    step: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a DataFrame into a 2D array of time-series windows.

    Parameters:
        - data: DataFrame with a 'value' column and an 'anomaly' column (1 if anomaly, 0 otherwise)
        - window_size: Size of the window
        - step: Step size between windows

    Returns:
        - 2D array of time-series windows
        - 1D array of anomaly labels
    """
    values = data["value"].values
    anomalies = data["anomaly"].values

    windows = np.array(
        [
            values[i : i + window_size]
            for i in range(0, len(values) - window_size + 1, step)
        ]
    )

    anomaly_labels = np.array(
        [
            1 if anomalies[i : i + window_size].sum() > 0 else 0
            for i in range(0, len(anomalies) - window_size + 1, step)
        ]
    )

    return windows, anomaly_labels
