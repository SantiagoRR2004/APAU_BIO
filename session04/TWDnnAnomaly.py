import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim

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


##############################################################################
# (B) Simple DNN Anomaly Detector in PyTorch
##############################################################################
class DNNAnomalyDetector(nn.Module):
    """
    A simple feed-forward neural network (fully connected) that
    classifies each window as normal (0) or anomaly (1).
    """

    def __init__(self, input_dim, hidden_dim=16, lr=1e-3, epochs=10, device="cpu"):
        super().__init__()

        # Define a small MLP for classification
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # output => 2 classes: normal vs anomaly
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.epochs = epochs
        self.device = device
        self.to(self.device)

        self.train_loss_history = []

    def forward(self, x):
        """
        Forward pass through the MLP.
        x should be a 2D tensor: [batch_size, input_dim].
        """
        return self.model(x)

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=16):
        """
        Train the network on (X_train, y_train).
        If X_val, y_val are provided, we can measure validation performance each epoch.
        """
        # Convert data to torch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        # Create minibatches (for small data, we can do full-batch)
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0

            for Xb, yb in dataloader:
                # Forward
                preds = self.forward(Xb)  # shape: [batch_size, 2]
                loss = self.criterion(preds, yb)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.train_loss_history.append(avg_loss)

            # Optional: Evaluate on validation
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate_accuracy(X_val, y_val)
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}")

    def evaluate_accuracy(self, X_val, y_val):
        """
        Utility to compute classification accuracy on a validation set.
        """
        self.eval()
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        with torch.no_grad():
            preds = self.forward(X_val_t)  # shape: [N, 2]
            pred_labels = preds.argmax(dim=1)
            accuracy = (pred_labels == y_val_t).float().mean().item()
        return accuracy

    def predict(self, X):
        """
        Predict class labels for X => returns a NumPy array of 0 or 1.
        """
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.forward(X_t)
            pred_labels = logits.argmax(dim=1)
        return pred_labels.cpu().numpy()


##############################################################################
# (C) Demo: Train & Evaluate the DNN
##############################################################################
if __name__ == "__main__":

    # 1) Generate data
    anomaly_intervals = [(100, 120), (250, 270)]
    T, X, y, window_starts = generate_single_timeseries_with_anomalies(
        n_points=400,
        anomaly_intervals=anomaly_intervals,
        window_size=100,
        step=20,
        random_seed=42,
    )

    # 2) Visualize data and windows
    plot_timeseries_with_windows(
        T,
        anomaly_intervals,
        window_size=100,
        window_starts=window_starts,
        y=y,
        title="Single Time Series with Marked Windows & Anomalies",
    )

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    # 4) Build and train DNN model
    input_dim = X_train.shape[1]  # length of each time window
    dnn_detector = DNNAnomalyDetector(
        input_dim=input_dim,
        hidden_dim=16,
        lr=1e-3,
        epochs=20,  # increase epochs if needed
        device="cpu",  # or "cuda" if you have a GPU
    )

    dnn_detector.fit(X_train, y_train, X_val=X_test, y_val=y_test, batch_size=16)

    # 5) Evaluate on test set
    y_pred = dnn_detector.predict(X_test)

    print("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    # 6) Plot training loss
    plt.figure()
    plt.plot(dnn_detector.train_loss_history, marker="o")
    plt.title("DNN Anomaly Detector: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # (Optional) Use PCA for a 2D visualization:
    pca = PCA(n_components=2).fit(X)
    X2_test = pca.transform(X_test)
    plt.figure()
    plt.title("Test Data Visualization (PCA 2D)")
    plt.scatter(
        X2_test[y_pred == 0, 0],
        X2_test[y_pred == 0, 1],
        c="blue",
        alpha=0.5,
        label="Pred Normal",
    )
    plt.scatter(
        X2_test[y_pred == 1, 0],
        X2_test[y_pred == 1, 1],
        c="red",
        alpha=0.5,
        label="Pred Anomaly",
    )
    # Mark true anomalies with black edges:
    anomaly_idx = np.where(y_test == 1)[0]
    plt.scatter(
        X2_test[anomaly_idx, 0],
        X2_test[anomaly_idx, 1],
        facecolors="none",
        edgecolors="black",
        s=80,
        label="True Anomaly",
    )
    plt.legend()
    plt.show()
