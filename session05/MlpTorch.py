import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.1):
        """
        Initializes the MLP with one hidden layer using PyTorch.
        """
        super(PyTorchMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # Input to Hidden Layer
        self.output = nn.Linear(hidden_size, output_size)  # Hidden to Output Layer
        self.activation = nn.Sigmoid()  # Sigmoid activation for non-linearity
        
        # Training settings
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        # Stochastic Gradient Descent
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)  

    def forward(self, x):
        """
        Forward pass: Computes the output of the network.
        """
        x = self.activation(self.hidden(x))  # Hidden layer activation
        x = self.activation(self.output(x))  # Output layer activation
        return x
        
        
    def train_model(self, X, y, epochs=5000):
        """
        Training function using backpropagation and gradient descent.
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Reset gradients
            y_pred = self.forward(X)  # Forward pass
            loss = self.criterion(y_pred, y)  # Compute loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Print progress every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, X):
        """
        Runs the network on new inputs and returns the predictions.
        """
        with torch.no_grad():
            predictions = self.forward(X)
        return predictions.round()
    
    
def main():
    """
    Main function to train the PyTorch MLP on the XOR problem and test its performance.
    """
    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # Instantiate and train the MLP using backpropagation
    mlp = PyTorchMLP()
    mlp.train_model(X, y)

    # Test predictions
    print("\nFinal Predictions:")
    predictions = mlp.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i].tolist()}, Predicted: {predictions[i].item()}, Expected: {y[i].item()}")
