# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model for production inference
model = joblib.load('aeropure_model.pkl')

def get_aqi_category(aqi):
    if aqi <= 50:
        return {"category": "Good", "color": "#2ecc71"}
    elif aqi <= 100:
        return {"category": "Moderate", "color": "#f1c40f"}
    elif aqi <= 200:
        return {"category": "Poor / Unhealthy", "color": "#e67e22"}
    else:
        return {"category": "Severe / Hazardous", "color": "#e74c3c"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        no2 = float(data['no2'])
        
        features = np.array([[pm25, pm10, no2]])
        prediction = model.predict(features)[0]
        aqi_val = round(float(prediction), 2)
        
        aqi_info = get_aqi_category(aqi_val)
        
        return jsonify({
            'status': 'success',
            'predicted_aqi': aqi_val,
            'category': aqi_info['category'],
            'color': aqi_info['color'],
            'model_accuracy': '98.73% (R² Score)'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)