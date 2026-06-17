import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

# Load dataset
df = pd.read_csv("data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Rename columns for simplicity
df.rename(columns={
    'Total_Water_Consumption_(Billion_Cubic_Meters)': 'Total_Water_Consumption',
    'Per_Capita_Water_Use_(Liters_per_Day)': 'Per_Capita_Use',
    'Agricultural_Water_Use_(%)': 'Agricultural_Use',
    'Industrial_Water_Use_(%)': 'Industrial_Use',
    'Household_Water_Use_(%)': 'Household_Use',
    'Rainfall_Impact_(Annual_Precipitation_in_mm)': 'Rainfall',
    'Groundwater_Depletion_Rate_(%)': 'Groundwater_Depletion',
    'Water_Scarcity_Level': 'Scarcity'
}, inplace=True)

# Encode categorical column
df['Scarcity'] = df['Scarcity'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

# One-hot encode Country
df = pd.get_dummies(df, columns=['Country'], drop_first=True)

# Features & target
X = df.drop('Total_Water_Consumption', axis=1)
y = df['Total_Water_Consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "columns.pkl")