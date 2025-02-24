from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
#autoencoder = load_model("./models/autoencoder.h5", custom_objects={'MeanSquaredError': MeanSquaredError()})
# Load autoencoder if available
#autoencoder_path = "./models/autoencoder.h5"
#if os.path.exists(autoencoder_path):
    #autoencoder = load_model(autoencoder_path, custom_objects={'MeanSquaredError': MeanSquaredError()})
#else:
    #print("⚠️ Warning: Autoencoder model not found!")

# Load LOF model if available
#lof_path = "models/lof_model.pkl"
#if os.path.exists(lof_path):
    #lof = joblib.load(lof_path)
#else:
    #print("⚠️ Warning: LOF model not found!")

autoencoder = load_model(
    "./models/autoencoder.h5",
    custom_objects={
        'mse': MeanSquaredError(),
        'MeanSquaredError': MeanSquaredError()
    }
)

lof = joblib.load("models/lof_model.pkl")

# Function to process uploaded file
def process_file(filepath):
    df = pd.read_csv(filepath)
    
    # Ensure only numerical features are used (Modify this as needed)
    X = df.select_dtypes(include=[np.number]).values  # Convert to NumPy array

    # Autoencoder predictions
    X_pred = autoencoder.predict(X)
    reconstruction_errors = np.mean(np.abs(X - X_pred), axis=1).reshape(-1, 1)
    
    # LOF predictions
    y_scores = lof.decision_function(reconstruction_errors)
    y_pred = lof.predict(reconstruction_errors)
    
    # Convert LOF predictions: -1 (anomaly) → 1, 1 (normal) → 0
    y_pred = np.where(y_pred == 1, 0, 1)  

    # Add predictions to dataframe
    df['Anomaly_Score'] = y_scores
    df['Prediction'] = y_pred  # ✅ This is the key column

    # Save processed data
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    df.to_csv(results_filepath, index=False)

    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        df = process_file(filepath)
        results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
        df.to_csv(results_filepath, index=False)
        return redirect(url_for('results', filename='results.csv'))
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('error'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("error.html", message="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template("error.html", message="No file selected.")

    try:
        # Read CSV
        df = pd.read_csv(file)
        print("File uploaded successfully. Shape:", df.shape)  # Debugging

        # Preprocessing
        X = df.values  # Assuming it's already in numerical format
        print("Preprocessed input shape:", X.shape)  # Debugging

        # Get reconstruction errors from the autoencoder
        X_pred = autoencoder.predict(X)
        reconstruction_errors = np.mean(np.abs(X - X_pred), axis=1).reshape(-1, 1)
        print("Reconstruction errors calculated.")  # Debugging

        # Get anomaly scores from LOF
        lof_scores = lof.decision_function(reconstruction_errors)
        predictions = (lof_scores < 0).astype(int)  # 1 = anomaly, 0 = normal
        print("Predictions computed:", predictions[:10])  # Debugging

        df['Anomaly'] = predictions
        return render_template("results.html", tables=[df.to_html()], titles=df.columns.values)

    except Exception as e:
        print("Error during prediction:", e)
        return render_template("error.html", message=f"Prediction error: {str(e)}")


@app.route('/results')
def results():
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    if not os.path.exists(results_filepath):
        flash("No results available. Please upload a file first.")
        return redirect(url_for('index'))

    df = pd.read_csv(results_filepath)

    return render_template('results.html', tables=[df.to_html(classes='table table-bordered table-striped', index=False)], titles=df.columns.values)



@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)



