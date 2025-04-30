# Define input variables (1 = yes, 0 = no)
X1 = 1  # Good waves
X2 = 0  # Empty lineup
X3 = 1  # Shark-free zone

# Define weights
W1 = 5  # Importance of good waves
W2 = 2  # Importance of less crowded lineup
W3 = 4  # Importance of avoiding sharks

# Define bias (threshold)
bias = -3

# Compute the output (Y-hat)
Y_hat = (X1 * W1) + (X2 * W2) + (X3 * W3) - bias

# Determine if the output is greater than 0
if Y_hat > 0:
    print("Output: 1 (We’re going surfing!)")
else:
    print("Output: 0 (No surfing today.)")

# Display the computed value
print(f"Y_hat: {Y_hat}")
