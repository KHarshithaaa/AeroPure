# train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Synthetic Dataset Generation
np.random.seed(42)
n_samples = 1000

pm25 = np.random.uniform(10, 250, n_samples)
pm10 = np.random.uniform(20, 350, n_samples)
no2 = np.random.uniform(5, 100, n_samples)

air_score = 0.5 * pm25 + 0.3 * pm10 + 0.2 * no2 + np.random.normal(0, 5, n_samples)

df = pd.DataFrame({
    'pm25': pm25,
    'pm10': pm10,
    'no2': no2,
    'air_score': air_score
})

csv_path = 'data/air_quality_data.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset generated and saved to {csv_path}")

# Model Training & Evaluation
X = df[['pm25', 'pm10', 'no2']]
y = df['air_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"Model Evaluation -> RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

# Save trained model
joblib.dump(model, 'aeropure_model.pkl')
print("Model saved successfully as aeropure_model.pkl!")