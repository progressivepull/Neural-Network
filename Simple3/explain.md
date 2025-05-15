Here's a simple movie recommendation system using neural networks built from scratch with NumPy—no TensorFlow or deep learning frameworks required.

# Overview:
* **Inputs:** User's genre preferences, past ratings.
* **Hidden Layers:**  Process the inputs to find patterns.
* **Output Layer:**  Suggest movies based on learned preferences.

```
import numpy as np

# Define sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample movie genres and user preferences
movie_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]
user_preferences = np.array([0.8, 0.2, 0.5, 0.7, 0.1])  # User preference scores (0 to 1)

# Define neural network structure
input_size = len(movie_genres)  # 5 input nodes (one per genre)
hidden_size = 6
output_size = 5  # Predict one score per genre

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)

weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

# Forward propagation
hidden_layer = sigmoid(np.dot(user_preferences, weights_input_hidden) + bias_hidden)
output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

# Recommend top 3 genres
recommended_indices = np.argsort(output_layer)[::-1][:3]
recommended_genres = [movie_genres[i] for i in recommended_indices]

# Display results
print("Top 3 recommended genres:", recommended_genres)

```

## Explanation:
* **1. Input Layer:** Takes user preferences (like/dislike for different genres).
* **2. Hidden Layer:** Processes patterns in preferences.
* **3. Output Layer:** Predicts how much the user will like each genre.
* **4. Recommendation:** Returns top 3 genres based on predictions.