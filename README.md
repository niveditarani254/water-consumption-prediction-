# 💧 AI-Powered Water Demand Forecasting Dashboard

## Overview

This project is a Machine Learning-based web application that predicts water demand using environmental, demographic, and economic factors.

The system uses a Random Forest Regression model and provides an interactive Streamlit dashboard for making predictions and visualizing results.

---

## Features

✅ Water demand prediction

✅ Interactive Streamlit dashboard

✅ Country-wise forecasting

✅ Feature importance visualization

✅ Machine Learning model using Random Forest

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## Project Structure
water-consumption-prediction/
│
├── app.py # Streamlit Dashboard
├── model.py # Model Training Script
├── model.pkl # Trained Model
├── columns.pkl # Feature Columns
├── data.csv # Dataset
├── requirements.txt # Dependencies
└── README.md


---

## Machine Learning Model

Algorithm Used:

- Random Forest Regressor

Input Features:

- Country
- Year
- Per Capita Water Use
- Agricultural Water Use
- Industrial Water Use
- Population
- Rainfall
- Temperature
- GDP

Output:

- Predicted Total Water Consumption

---

## Installation

Clone the repository:

```bash
git clone https://github.com/niveditarani254/water-consumption-prediction-.git
cd water-consumption-prediction-

Dashboard Preview

The dashboard allows users to:

Enter water consumption factors
Generate demand predictions
View prediction results instantly
Explore feature importance analysis


Future Improvements
Real-time weather integration
Time-series forecasting
Advanced visualization dashboards
Cloud deployment support


Author
Nivedita Rani
