from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# Paths
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'
cols_path = 'model_columns.pkl'

# Global variables
model = None
scaler = None
model_cols = None

def load_artifacts():
    global model, scaler, model_cols
    
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        if os.path.exists(cols_path):
            with open(cols_path, 'rb') as f:
                model_cols = pickle.load(f)

    except Exception as e:
        print("Error loading artifacts:")
        print(traceback.format_exc())

# Load once at startup
load_artifacts()

@app.route('/')
def home():
    return render_template('index.html', features=model_cols)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or model_cols is None:
        return jsonify({
            'error': 'Model artifacts not found. Please ensure model_training.py has been executed.'
        }), 400

    try:
        features = []

        for col in model_cols:
            val = request.form.get(col)

            if val is None or val.strip() == '':
                return jsonify({
                    'error': f"Missing value for: {col.replace('_', ' ').title()}"
                }), 400

            try:
                features.append(float(val))
            except ValueError:
                return jsonify({
                    'error': f"Invalid numeric value for: {col.replace('_', ' ').title()}"
                }), 400

        # Apply log transformation if required
        if 'policy_annual_premium' in model_cols:
            idx = model_cols.index('policy_annual_premium')
            features[idx] = np.log1p(features[idx])

        final_features = np.array(features).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)

        prediction = model.predict(final_features_scaled)

        if prediction[0] == 1:
            result = "Fraud Detected"
            status = "fraud"
        else:
            result = "Genuine Claim"
            status = "genuine"

        return jsonify({
            "prediction": result,
            "status": status
        })

    except Exception:
        print("Prediction error:")
        print(traceback.format_exc())
        return jsonify({
            "error": "Internal server error during prediction."
        }), 500


if __name__ == "__main__":
    # Production-safe run configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
