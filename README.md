# water-consumption-prediction-
# AI-Powered Water Demand Forecasting Dashboard

## Overview

This project is a machine learning-based web application that predicts water demand using environmental, demographic, and economic factors. The application uses a Random Forest Regression model and provides an interactive dashboard built with Streamlit.

## Features

* Predicts total water consumption based on user inputs.
* Interactive and user-friendly dashboard.
* Machine Learning model built using Random Forest Regression.
* Real-time prediction generation.
* Data visualization for better analysis.

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Joblib

## Project Structure

water-prediction/

├── app.py              # Streamlit application

├── model.py            # Model training script

├── model.pkl           # Trained ML model

├── columns.pkl         # Feature columns

├── data.csv            # Dataset

├── requirements.txt    # Dependencies

└── README.md           # Documentation

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd water-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run app.py
```

## Machine Learning Model

The project uses a Random Forest Regression model trained on historical water consumption data. The model learns relationships between multiple factors and predicts future water demand.

## Applications

* Smart City Planning
* Water Resource Management
* Municipal Water Supply Forecasting
* Sustainability and Conservation Analysis

## Future Improvements

* Integration with real-time weather APIs
* Advanced forecasting using Deep Learning
* Interactive analytics dashboard
* Deployment on Streamlit Cloud or AWS

## Author

Developed as a Machine Learning project for water demand forecasting and resource optimization.
