from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Get the absolute path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load the model
model_path = os.path.join(models_dir, 'linear_regression_model.joblib')
model = joblib.load(model_path)

# Load other variables
variables_path = os.path.join(models_dir, 'model_variables.joblib')
variables = joblib.load(variables_path)

x_train = variables['x_train']
x_test = variables['x_test']
y_train = variables['y_train']
y_test = variables['y_test']
slope = variables['slope']
intercept = variables['intercept']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        cgpa = float(request.form['cgpa'])

        # Make a prediction using the loaded model
        prediction = model.predict(np.array([[cgpa]]))

        rounded_prediction = round(prediction[0][0], 2)

        return render_template('result.html', prediction=rounded_prediction)

if __name__ == '__main__':
    app.run(debug=True)
