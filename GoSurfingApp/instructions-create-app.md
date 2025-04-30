Here's a basic Python Flask web app to fulfill your request. Users can input values for x1, x2, and x3. The "Training" button generates random weights (w1, w2, w3) and calculates the output message based on the formula.

**Steps to Run the App:**
1. Install Flask: pip install flask   

2. Save the following code into a file, e.g., app.py.   

3. Run the script: python app.py   

4. Open your browser and navigate to C.

```python
from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Initialize global weights
weights = {"w1": 0, "w2": 0, "w3": 0}

@app.route("/", methods=["GET", "POST"])
def home():
    output_message = ""
    y_hat = 0

    if request.method == "POST":
        if "train" in request.form:
            # Generate random weights
            weights["w1"] = random.randint(1, 5)
            weights["w2"] = random.randint(1, 5)
            weights["w3"] = random.randint(1, 5)
            output_message = "Weights updated: w1={}, w2={}, w3={}".format(
                weights["w1"], weights["w2"], weights["w3"]
            )

        elif "predict" in request.form:
            # Get input values from the form
            x1 = int(request.form.get("x1", 0))
            x2 = int(request.form.get("x2", 0))
            x3 = int(request.form.get("x3", 0))
            bias = -3

            # Calculate Y-hat
            y_hat = (x1 * weights["w1"]) + (x2 * weights["w2"]) + (x3 * weights["w3"]) - bias

            # Determine the output message
            if y_hat > 0:
                output_message = f"Y-hat: {y_hat}. Output: 1 (We’re going surfing!)"
            else:
                output_message = f"Y-hat: {y_hat}. Output: 0 (No surfing today.)"

    return render_template("index.html", output_message=output_message, weights=weights)

if __name__ == "__main__":
    app.run(debug=True)
```
Create a file templates/index.html with the following content:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Surf Decision App</title>
</head>
<body>
    <h1>Surf Decision App</h1>
    <form method="POST">
        <label for="x1">Good waves (x1):</label>
        <input type="number" id="x1" name="x1" required>
        <br>
        <label for="x2">Empty lineup (x2):</label>
        <input type="number" id="x2" name="x2" required>
        <br>
        <label for="x3">Shark-free zone (x3):</label>
        <input type="number" id="x3" name="x3" required>
        <br><br>
        <button type="submit" name="train">Training (Generate Random Weights)</button>
        <button type="submit" name="predict">Predict (Surf or Not)</button>
    </form>
    <h3>{{ output_message }}</h3>
    <p>Current Weights: w1={{ weights['w1'] }}, w2={{ weights['w2'] }}, w3={{ weights['w3'] }}</p>
</body>
</html>
```
This web app allows users to:       

1. Input values for x1, x2, and x3.    

2. Click "Training" to randomly generate new weights (w1, w2, and w3).    

3. Click "Predict" to calculate and display the output message.     



