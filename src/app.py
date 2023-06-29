from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the trained model
with open('src/models/bebum.pkl', 'rb') as f:
    knn_model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the form
    age = int(request.form['age'])
    parent_status = request.form['parent-status']
    mother_job = request.form['mother-job']
    father_job = request.form['father-job']
    guardian = request.form['guardian']
    study_time = int(request.form['study-time'])
    failures = int(request.form['failures'])
    internet = request.form['internet']
    romantic = request.form['romantic']
    famrel = int(request.form['famrel'])
    freetime = int(request.form['freetime'])
    goout = int(request.form['goout'])
    dalc = int(request.form['dalc'])
    walc = int(request.form['walc'])
    absences = int(request.form['abseneces'])
    g3 = int(request.form['g3'])

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'Pstatus': [parent_status],
        'Mjob': [mother_job],
        'Fjob': [father_job],
        'guardian': [guardian],
        'studytime': [study_time],
        'failures': [failures],
        'internet': [internet],
        'romantic': [romantic],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'Dalc': [dalc],
        'Walc': [walc],
        'absences': [absences],
        'G3': [g3]
    })


    # Perform one-hot encoding on the input data
    input_data_encoded = pd.get_dummies(input_data)

    # Make predictions using the loaded model
    prediction = knn_model.predict(input_data_encoded)

    # Pass the prediction result to the template
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)