from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Make prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Map the prediction to the class label
        iris_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_class = iris_species[prediction[0]]
        
        # Render result on HTML page
        return render_template("index.html", prediction_text=f'The predicted Iris flower species is: {predicted_class}')
    
    except Exception as e:
        return render_template("index.html", prediction_text="Error occurred: Please ensure all inputs are correct.")

if __name__ == "__main__":
    app.run(debug=True)