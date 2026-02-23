import os
import numpy as np
import joblib
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'diabetes_prediction_secret_key'

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Load Model and Scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Models loaded successfully.")
except FileNotFoundError:
    model = None
    scaler = None

# Input Validation Ranges (Based on medical standards/dataset limits)
VALIDATION_RULES = {
    'Pregnancies': (0, 20),
    'Glucose': (50, 200),
    'BloodPressure': (40, 150),
    'SkinThickness': (0, 100),
    'Insulin': (0, 900),
    'BMI': (10, 70),
    'DiabetesPedigreeFunction': (0.0, 3.0),
    'Age': (18, 100)
}

def validate_input(data):
    errors = []
    for key, (min_val, max_val) in VALIDATION_RULES.items():
        try:
            val = float(data.get(key))
            if not (min_val <= val <= max_val):
                errors.append(f"{key} must be between {min_val} and {max_val}.")
        except (TypeError, ValueError):
            errors.append(f"Invalid value for {key}.")
    return errors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('error.html', message="Models not found. Please run train.py first.")

    try:
        # Get input data
        input_data = request.form
        
        # Validate Input
        errors = validate_input(input_data)
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('index'))

        # Prepare features for model
        features = np.array([[ 
            float(input_data['Pregnancies']),
            float(input_data['Glucose']),
            float(input_data['BloodPressure']),
            float(input_data['SkinThickness']),
            float(input_data['Insulin']),
            float(input_data['BMI']),
            float(input_data['DiabetesPedigreeFunction']),
            float(input_data['Age'])
        ]])

        # Scale and Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100

        result = "Positive (Diabetes Risk)" if prediction == 1 else "Negative (No Diabetes Risk)"
        
        return render_template('result.html', 
                               result=result, 
                               probability=round(probability, 2),
                               input_data=input_data)

    except Exception as e:
        flash(f"An error occurred: {str(e