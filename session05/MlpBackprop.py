import numpy as np

class MLPBackprop:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.1):
        """
        Initializes the MLP with randomly assigned weights and biases.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weight matrices with random values
        self.W_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.b_hidden = np.random.randn(self.hidden_size)

        self.W_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.b_output = np.random.randn(self.output_size)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    def forward(self, x):
        """
        Forward pass: Computes the output of the network.
        Returns hidden layer output and final output.
        """
        hidden_layer_input = np.dot(x, self.W_input_hidden) + self.b_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        final_output_input = np.dot(hidden_layer_output, self.W_hidden_output) + self.b_output
        final_output = self.sigmoid(final_output_input)

        return hidden_layer_output, final_output
    
    def train(self, X, y, epochs=5000):
        """
        Training loop using backpropagation and gradient descent.
        """
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                hidden_layer_output, final_output = self.forward(X[i])

                # Compute error
                error = y[i] - final_output
                total_error += np.sum(error**2) / 2  # Mean squared error

                # Backpropagation: Compute gradients
                delta_output = error * self.sigmoid_derivative(final_output)
                delta_hidden = (delta_output @ self.W_hidden_output.T) * self.sigmoid_derivative(hidden_layer_output)

                # Gradient Descent Weight Updates
                self.W_hidden_output += self.learning_rate * np.outer(hidden_layer_output, delta_output)
                self.b_output += self.learning_rate * delta_output

                self.W_input_hidden += self.learning_rate * np.outer(X[i], delta_hidden)
                self.b_hidden += self.learning_rate * delta_hidden

            # Print progress every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")

    def predict(self, X):
        """
        Runs the network on new inputs and returns the predictions.
        """
        predictions = []
        for i in range(len(X)):
            _, final_output = self.forward(X[i])
            predictions.append(final_output[0])  # Convert to scalar value
        return np.round(predictions, 2)
        
    
    
def main():
    """
    Main function to train the MLP on the XOR problem and test its performance.
    """
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # Expected XOR outputs (reshaped for matrix operations)

    # Instantiate and train the MLP using backpropagation
    mlp = MLPBackprop()
    mlp.train(X, y)

    # Test predictions
    print("\nFinal Predictions:")
    predictions = mlp.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i]}, Expected: {y[i][0]}")
