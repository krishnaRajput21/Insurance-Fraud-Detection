# Automobile Insurance Fraud Detection System

This is an end-to-end Machine Learning project designed to detect automobile insurance fraud, built with Python, scikit-learn, and Flask.

## Project Structure

- `model_training.py`: Generates a fully synthetic and realistic dataset of 1000 claims. Perform data preprocessing (outlier capping via IQR, log transformation, dropping highly correlated columns), trains 6 different Machine Learning classifiers, and saves the best performing model (`model.pkl`) using Pickle.
- `app.py`: The Flask web application backend. It loads the created machine learning model, scaler, and the required features, serving the frontend.
- `templates/index.html`: A modern, clean HTML/CSS web frontend with a form to input automobile claim details to get fraud predictions.
- `requirements.txt`: Follows all package dependencies required for the project.
- `setup.bat`: An automated Windows batch script to install requirements, run data training, and start the app.

## How to Run the Project

### Option 1: One-Click Start (Windows)
1. Double-click the `setup.bat` file in your File Explorer. 
2. It will automatically:
   - Install the required dependencies.
   - Run `model_training.py` to generate the dataset and the required `.pkl` model files.
   - Start the Flask web server (`app.py`).
3. Once the server starts in the terminal, open your web browser and go to `http://127.0.0.1:5000`

### Option 2: Manual Setup (Terminal)
If you prefer to run the commands manually or are using a different operating system (macOS/Linux), open your terminal/command prompt in this project folder and follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data and Train Model:**
   ```bash
   python model_training.py
   ```
   *(This step creates `insurance_claims.csv`, `model.pkl`, `scaler.pkl`, and `model_columns.pkl`)*

3. **Start the Web Server:**
   ```bash
   python app.py
   ```

4. **Access the Web App:**
   Open your browser and navigate to `http://127.0.0.1:5000`
