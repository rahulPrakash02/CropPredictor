import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the pickled model from file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create a Flask web application
app = Flask(__name__)
top_3_crops = []
prices = {
    'rice': 2000,
    'maize': 1500,
    'chickpea': 4000,
    'kidneybeans': 3000,
    'pigeonpeas': 3500,
    'mothbeans': 3800,
    'mungbean': 3200,
    'blackgram': 4200,
    'lentil': 4500,
    'pomegranate': 2500,
    'banana': 1800,
    'mango': 3000,
    'grapes': 2800,
    'watermelon': 1200,
    'muskmelon': 1400,
    'apple': 3500,
    'orange': 2200,
    'papaya': 1600,
    'coconut': 4000,
    'cotton': 4500,
    'jute': 3200,
    'coffee': 5000
}


# Define the home route
@app.route('/')
def home():
    return render_template('index.html', prediction = top_3_crops, dict = prices)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    nitrogen = request.form['nitrogen']
    phosphorus = request.form['phosphorus']
    potassium = request.form['potassium']
    temperature = request.form['temperature']
    humidity = request.form['humidity']
    ph = request.form['ph']
    rainfall = request.form['rainfall']

    # Make a prediction using the loaded model
    features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    probabilities = loaded_model.predict_proba(features)

    
    for i in range(len(probabilities)):
        top_3_indices = probabilities[i].argsort()[::-1][:3]
        top_3_crops.append(loaded_model.classes_[top_3_indices])
    
    print(top_3_crops)

    # Render the prediction result on a template
    return render_template('index.html', prediction = top_3_crops, dict = prices)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
