from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('rf_acc_68.pkl')
scaler = joblib.load('normalizer.pkl')

# Show class labels in console
print("Model class labels:", model.classes_)  # This will print [1 2]

# Feature order
feature_columns = ['age', 'gender', 'tot_bilirubin', 'direct_bilirubin',
                   'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Safely extract form data
        form_values = [request.form.get(key) for key in feature_columns]
        if None in form_values:
            return jsonify({"error": "Missing form data"}), 400

        data = [float(x) for x in form_values]

        # Create dataframe and scale it
        df = pd.DataFrame([data], columns=feature_columns)
        scaled_data = scaler.transform(df)

        # Predict probabilities
        proba = model.predict_proba(scaled_data)[0]
        proba_dict = dict(zip(model.classes_, proba))

        # Get correct confidence values
        healthy_conf = round(proba_dict[1] * 100, 2)  # class 1 = healthy
        diseased_conf = round(proba_dict[2] * 100, 2)  # class 2 = diseased

        print(
            f"🧠 Confidence → Healthy: {healthy_conf}%, Diseased: {diseased_conf}%")

        # Decision: if disease probability > healthy → risky
        status = 1 if diseased_conf > healthy_conf else 0

        return jsonify({
            "redirect": url_for("result", status=status, healthy_conf=healthy_conf, diseased_conf=diseased_conf)
        })

    except Exception as e:
        print("Prediction error occurred:")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/result")
def result():
    status = request.args.get("status")
    healthy_conf = request.args.get("healthy_conf")
    diseased_conf = request.args.get("diseased_conf")

    if status == "1":
        message = "⚠️ Risk of Liver Disease detected. Please consult a doctor."
        advice = "Limit alcohol, eat liver-friendly foods, and consult a hepatologist immediately."
        risk_level = "danger"
    elif status == "0":
        message = "✅ You are safe!"
        advice = "Stay hydrated, eat a balanced diet, and go for routine checkups."
        risk_level = "safe"
    else:
        return redirect(url_for('index'))

    return render_template("result.html", message=message, advice=advice,
                           healthy_conf=healthy_conf, diseased_conf=diseased_conf, risk_level=risk_level)


if __name__ == '__main__':
    app.run(debug=True)
