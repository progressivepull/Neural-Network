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

