💳 AI-Powered Financial Fraud Detection System

<p align="center">
  <b>Machine Learning based financial transaction fraud detection with an interactive Streamlit dashboard.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

📌 Overview

The AI-Powered Financial Fraud Detection System is a machine learning application that analyzes financial transaction details and predicts whether a transaction is potentially fraudulent.

The project combines a trained Logistic Regression pipeline with an interactive Streamlit web application. Users can enter transaction information, receive a fraud probability and risk level, and explore transaction analytics through an interactive dashboard.

The goal is to make fraud detection easier to understand and demonstrate how machine learning can support financial security and faster decision-making.

✨ Features

🔍 Real-time transaction prediction

🎯 Fraud probability / risk score

🚨 LOW, MEDIUM and HIGH risk classification

📊 Interactive fraud analytics dashboard

📋 Transaction monitoring and filtering

🔎 Transaction ID search

⬇️ Download filtered transaction results as CSV

🧠 Scikit-learn machine learning pipeline

⚡ Streamlit interactive frontend

📈 Plotly visualizations

🛡️ Preprocessing included inside the trained pipeline

🧠 Machine Learning

The project uses a trained Logistic Regression classifier with preprocessing handled through a Scikit-learn pipeline.

Pipeline

Transaction Input
       ↓
Data Preprocessing
       ↓
StandardScaler
       +
OneHotEncoder
       ↓
Logistic Regression
       ↓
Fraud Prediction
       ↓
Fraud Probability
       ↓
Risk Level

Input Features

The prediction interface uses:

Feature

Description

step

Transaction time step

type

Transaction type

amount

Transaction amount

oldbalanceOrg

Sender balance before transaction

newbalanceOrig

Sender balance after transaction

oldbalanceDest

Receiver balance before transaction

newbalanceDest

Receiver balance after transaction

Transaction Types

CASH_IN

CASH_OUT

DEBIT

PAYMENT

TRANSFER

The Streamlit interface converts the human-readable transaction type into the numeric representation expected by the trained model.

🖥️ Application Pages

🏠 Home

Provides an overview of the system along with:

Total transactions

Fraud cases

Fraud rate

Transaction volume

System workflow

Transaction activity chart

🔍 Predict

Users enter transaction information and receive:

Fraud probability

Prediction result

Risk level

Risk progress bar

Transaction summary

Example:

Fraud Probability: 82.45%
Risk Level: HIGH
Prediction: FRAUD

📊 Dashboard

Provides interactive analytics including:

Fraud vs legitimate transactions

Transactions by type

Fraud cases by transaction type

Transaction amount distribution

Fraud probability distribution

📋 Transactions

Allows users to:

Filter by fraud status

Filter by transaction type

Search transaction IDs

View transaction details

Download filtered records

ℹ️ About

Provides information about the project, technologies and machine learning pipeline.

🛠️ Tech Stack

Technology

Purpose

Python

Core development

Pandas

Data manipulation

NumPy

Numerical operations

Scikit-learn

Machine learning & preprocessing

Joblib

Model serialization/loading

Plotly

Interactive visualizations

Streamlit

Web application/dashboard

Git & GitHub

Version control

📂 Project Structure

Financial-Fraud-Detection/
│
├── app.py
├── Fraud_detection_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore

Note: Keep the trained .pkl file in the same directory as app.py if the application is loading it locally.

⚙️ Installation

1. Clone the repository

git clone https://github.com/sanskritika2409/AI-Powered-Financial-Fraud-Detection-System
cd YOUR_REPOSITORY

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

The application will open in your browser.

📦 Requirements

The project requires:

streamlit
pandas
numpy
plotly
scikit-learn
joblib

Install them with:

pip install -r requirements.txt

🔐 Model File

The application expects the trained model pipeline to be named:

Fraud_detection_pipeline.pkl

and placed next to:

app.py

If the model is not found, the application will display a model-status warning instead of silently failing.

⚠️ GitHub File Size Note

GitHub has file-size limitations for normal repository files. If your .pkl is large, use Git LFS rather than committing a large binary directly.

For Git LFS:

git lfs install
git lfs track "*.pkl"
git add .gitattributes

Then commit and push normally.

📊 Risk Classification

The application converts the model's fraud probability into an easy-to-understand risk level:

Probability

Risk

< 40%

🟢 LOW

40% – 69.99%

🟡 MEDIUM

≥ 70%

🔴 HIGH

These thresholds are application-level presentation thresholds and can be adjusted according to the project's evaluation requirements.

🚀 Future Improvements

Connect the dashboard to a live transaction database

Add user authentication

Add email/SMS fraud alerts

Add model comparison with Random Forest and XGBoost

Add model evaluation page

Add confusion matrix and ROC-AUC visualization

Add real-time transaction streaming

Add explainable AI / feature importance

Deploy the application to Streamlit Community Cloud

Add automated model retraining

🎯 Project Objective

The main objective is to demonstrate how machine learning can be applied to financial transaction data to identify potentially fraudulent activity and present the results through a simple, human-friendly interface.

"Detect suspicious activity early, understand the risk, and make faster decisions."

👩‍💻 Author

Sanskritika Awasthi

Computer Science Engineering | Data Science & AI

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
