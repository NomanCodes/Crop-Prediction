from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ✅ Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''

    if request.method == 'POST':
        try:
            nitrogen = float(request.form['Nitrogen'])
            phosphorus = float(request.form['Phosphorus'])
            potassium = float(request.form['Potassium'])
            temperature = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['Rainfall'])

            features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]

            # ✅ Now make real prediction
            prediction = model.predict(features)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)
