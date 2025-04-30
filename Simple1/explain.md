# Explain Simple 1

Creating a simple neural network in Python requires defining the structure of the network, including the inputs, weights, and activation functions. Here's an example of a simple neural network with a single layer, basic forward propagation, and manual weight initialization:

# Python Code: Simple Neural Network with Weights and Inputs
```
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

```
# Explanation:
* **sigmoid():** This is the activation function used for both the hidden and output layers.

* **sigmoid_derivative():** This function calculates the derivative of the sigmoid function, which is required for the backpropagation step.

* **NeuralNetwork class:**

* The class defines a simple neural network with one hidden layer and an output layer.

* It contains methods for forward propagation (forward), backpropagation (backward), and training the network (train).

* The weights and biases are initialized randomly.

* During training, the network performs forward and backward passes for each epoch, adjusting the weights based on the errors computed by backpropagation.

# Training Process:
* Weights are adjusted during each backward pass based on the error between the predicted output (self.final_output) and the true output (y).

* The network is trained for 10,000 epochs with a learning rate of 0.1. The loss (mean squared error) is printed every 1000 epochs.

# Expected Output:
This neural network will attempt to learn the XOR function, where the inputs are 3 features and the output is a binary class (0 or 1). The output will improve as training progresses.
