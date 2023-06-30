from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
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
    absences = int(request.form['absences'])
    g3 = int(request.form['g3'])

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [study_time],
        'failures': [failures],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'Dalc': [dalc],
        'Walc': [walc],
        'absences': [absences],
        'G3': [g3]
    })

    not_int_input = pd.DataFrame({
        'Pstatus': [parent_status],
        'Mjob': [mother_job],
        'Fjob': [father_job],
        'guardian': [guardian],
        'internet': [internet],
        'romantic': [romantic],
    })

    input_data_encoded = pd.get_dummies(data=not_int_input)

    expected_cols = ['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'absences', 'G3', 'Pstatus_A', 'Pstatus_T', 'Mjob_at_home',
       'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
       'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services',
       'Fjob_teacher', 'guardian_father', 'guardian_mother', 'guardian_other',
       'internet_no', 'internet_yes', 'romantic_no', 'romantic_yes']
    
    input_data_final = pd.concat([input_data, input_data_encoded], axis=1)

    missing_cols = set(expected_cols) - set(input_data_final.columns)
    for col in missing_cols:
        input_data_final[col] = 0

    input_data_final = input_data_final[expected_cols]

    # Make predictions using the loaded model
    prediction = knn_model.predict(input_data_final)

    # Pass the prediction result to the template
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)