from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Path to your model file
model_path = 'Heart attack.pkl'

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trtbps = float(request.form['trtbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            exng = int(request.form['exng'])
            oldpeak = float(request.form['oldpeak'])
            slp = int(request.form['slp'])
            caa = int(request.form['caa'])
            thall = int(request.form['thall'])
            
            # Create an input array for the model (excluding thalachh)
            input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, exng, oldpeak, slp, caa, thall]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Create a prediction text based on the prediction result
            if prediction == 1:
                prediction_text = "The patient is likely to have heart disease."
            else:
                prediction_text = "The patient is unlikely to have heart disease."

            return render_template('index.html', prediction_text=prediction_text)

        except ValueError as e:
            return f"Error: {str(e)}"
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
