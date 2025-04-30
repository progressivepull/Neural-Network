# Explain Simple 2 

```
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean squared error loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate some dummy data
# Features (X): 100 samples, each with 3 features
# Targets (y): Binary (0 or 1)
np.random.seed(0)
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, size=(100, 1))

# Define the neural network structure
input_neurons = 3  # Number of features in the input
hidden_neurons = 5 # Number of neurons in the hidden layer
output_neurons = 1 # Binary output (0 or 1)

# Initialize weights and biases randomly
weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
bias_hidden = np.random.rand(1, hidden_neurons)
bias_output = np.random.rand(1, output_neurons)

# Training parameters
learning_rate = 0.1
epochs = 1000

# Training the network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    
    # Compute loss
    loss = mse_loss(y, output)
    
    # Backward pass
    error_output = y - output
    delta_output = error_output * sigmoid_derivative(output)
    
    error_hidden = delta_output.dot(weights_hidden_output.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Final output after training
print("\nTraining complete.")

```

# Explanation:
## 1. Structure:

* **Input Layer:** Takes the features as inputs (e.g., 3 features in this example).

* **Hidden Layer:** Contains 5 neurons with a sigmoid activation function.

* **Output Layer:** Outputs a single value (0 or 1) using a sigmoid activation function for binary classification.

## Training:

* The network uses forward propagation to calculate predictions.

* It computes the loss using Mean Squared Error (MSE).

* Then, it uses backpropagation to adjust the weights and biases by calculating gradients.

## Customization:

* You can modify the number of layers, neurons, activation functions, or training parameters like learning rate and epochs.

This approach is a low-level implementation to illustrate the inner workings of neural networks. For real-world projects, using libraries like TensorFlow or PyTorch is recommended for efficiency and scalability. Let me know if you'd like help extending this code!
