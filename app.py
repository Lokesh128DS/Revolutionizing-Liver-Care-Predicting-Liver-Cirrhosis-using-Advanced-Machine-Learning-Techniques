from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('liver_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = 'Liver Disease Detected' if prediction[0] == 1 else 'No Liver Disease Detected'
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
