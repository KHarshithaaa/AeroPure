# AeroPure - AI Air Quality Prediction & Analysis Platform

A production-ready full-stack AI application designed to preprocess environmental metrics, train supervised machine learning models, and serve real-time predictions via a Flask REST API and web user interface.

## 🚀 Tech Stack
* **Programming Language:** Python[cite: 2]
* **Data Processing & ML:** NumPy, pandas, scikit-learn[cite: 2], joblib
* **Backend & API:** Flask, REST APIs[cite: 2]
* **Frontend:** HTML5, CSS3, JavaScript
* **Version Control:** Git & GitHub[cite: 2]

## 🛠️ Project Architecture
1. **Data Preprocessing & Synthetic Generation:** Generates robust environmental metrics (`PM2.5`, `PM10`, `NO2`) and preprocesses them for ML modeling.
2. **Model Training & Evaluation:** Implements supervised machine learning (`LinearRegression`), evaluating performance metrics like **RMSE** and **R2 Score**, and serializing the model via `joblib` (`aeropure_model.pkl`).
3. **Production REST API & UI:** Serves real-time inference via Flask endpoints connected to an interactive frontend UI.

## ⚙️ Installation & Running Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/KHarshithaaa/AeroPure.git](https://github.com/KHarshithaaa/AeroPure.git)
   cd AeroPure