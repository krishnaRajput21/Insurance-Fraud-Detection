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

# Load artifacts
model = None
scaler = None
model_cols = None

def load_artifacts():
    global model, scaler, model_cols
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    if os.path.exists(cols_path):
        with open(cols_path, 'rb') as f:
            model_cols = pickle.load(f)

# Initial load
load_artifacts()

@app.route('/')
def home():
    # Attempt to reload in case model was trained after server start
    load_artifacts()
    return render_template('index.html', features=model_cols)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not model_cols:
        return jsonify({'error': 'Model not trained yet! Please run model_training.py first.'}), 400
        
    try:
        # Extract features from form in the EXACT order dictacted by model_cols
        features = []
        for col in model_cols:
            val = request.form.get(col)
            if val is None or val.strip() == '':
                return jsonify({'error': f"Missing value for required feature: {col.replace('_', ' ').capitalize()}"}), 400
                
            try:
                features.append(float(val))
            except ValueError:
                return jsonify({'error': f"Invalid input for {col.replace('_', ' ').capitalize()}. Must be a number."}), 400
            
        # Transform data
        # Check if policy_annual_premium needs the log1p transformation applied during training
        if 'policy_annual_premium' in model_cols:
            idx = model_cols.index('policy_annual_premium')
            features[idx] = np.log1p(features[idx])
            
        final_features = np.array(features).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)
        
        # Predict
        prediction = model.predict(final_features_scaled)
        
        if prediction[0] == 1:
            res = "Fraud Detected"
            status = 'fraud'
        else:
            res = "Genuine Claim"
            status = 'genuine'
            
        return jsonify({
            'prediction': res,
            'status': status
        })
        
    except Exception as e:
        # Print actual error to console for debugging
        print(f"Error making prediction: \n{traceback.format_exc()}")
        return jsonify({'error': 'An internal server error occurred while processing your request.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
