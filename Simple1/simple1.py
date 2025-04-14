import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)  # Weights from input to hidden layer
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)  # Weights from hidden to output layer

        # Initialize biases
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, learning_rate):
        # Backpropagation
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Train the neural network
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))  # Mean squared error loss
                print(f'Epoch {epoch} Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # Input data: 4 samples, 3 features each
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # Output data: corresponding labels
    y = np.array([[0], [1], [1], [0]])

    # Define the neural network
    input_size = 3  # Number of input features
    hidden_size = 4  # Number of hidden units
    output_size = 1  # Output layer size (for binary classification)

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the neural network
    print("Final output after training:")
    print(nn.forward(X))
