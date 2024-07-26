from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('rfc.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("output.html")

@app.route('/submit', methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        # Read the inputs given by the user
        animal_name = request.form['animalName']
        symptoms = [
            request.form['symptoms1'],
            request.form['symptoms2'],
            request.form['symptoms3'],
            request.form['symptoms4'],
            request.form['symptoms5']
        ]
        
        # Add a default value for the missing feature
        default_value = 0  # You can choose a different default value if needed
        while len(symptoms) < 6:
            symptoms.append(default_value)
        
        # Convert inputs to appropriate format and preprocess if necessary
        input_features = np.array(symptoms).reshape(1, -1)
        
        # Scale the features using the loaded scaler
        input_features = scaler.transform(input_features)
        
        # Predict using the loaded model
        prediction = model.predict(input_features)
        
        result = "Your health is in normal condition." if prediction == 1 else "According to our study, we feel sad."
        
        return render_template("output.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
