import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the dataset
heart_data = pd.read_csv('data.csv')

# Preprocess the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model from the file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get input data from the form and convert to numeric types
    input_data = []
    input_data.append(float(request.form['age']))
    input_data.append(float(request.form['sex']))
    input_data.append(float(request.form['cp']))
    input_data.append(float(request.form['trestbps']))
    input_data.append(float(request.form['chol']))
    input_data.append(float(request.form['fbs']))
    input_data.append(float(request.form['restecg']))
    input_data.append(float(request.form['thalach']))
    input_data.append(float(request.form['exang']))
    input_data.append(float(request.form['oldpeak']))
    input_data.append(float(request.form['slope']))
    input_data.append(float(request.form['ca']))
    input_data.append(float(request.form['thal']))

    # Convert input data to numpy array and reshape it
    input_data = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Determine prediction message
    if prediction[0] == 0:
        result = 'The person does not have a Heart Disease'
    else:
        result = 'The person has Heart Disease'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)