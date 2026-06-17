import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model & columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(
    page_title="AI-Powered Water Demand Forecasting Dashboard",
    page_icon="💧",
    layout="wide"
)


# ================= HEADER =================
st.title("💧 AI-Powered Water Demand Forecasting Dashboard")
st.markdown("""
This system predicts **total water consumption** based on environmental,
demographic, and economic factors using a **Random Forest model**.
""")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Input Parameters")

country = st.sidebar.selectbox("🌍 Country", ["India", "USA", "China", "Brazil", "Germany"])
year = st.sidebar.slider("📅 Year", 2000, 2030, 2022)
per_capita = st.sidebar.slider("👤 Per Capita Use (L/day)", 50, 500, 200)
agri = st.sidebar.slider("🌾 Agricultural Use (%)", 10, 90, 60)
ind = st.sidebar.slider("🏭 Industrial Use (%)", 5, 70, 25)
house = st.sidebar.slider("🏠 Household Use (%)", 5, 40, 15)
rain = st.sidebar.slider("🌧 Rainfall (mm)", 0, 3000, 1000)
ground = st.sidebar.slider("💧 Groundwater Depletion (%)", 0, 100, 30)
scarcity = st.sidebar.selectbox("⚠️ Water Scarcity Level", ["Low", "Medium", "High"])

# Encode scarcity
scarcity_map = {"Low": 0, "Medium": 1, "High": 2}

# ================= INPUT DATA =================
input_data = {
    'Year': year,
    'Per_Capita_Use': per_capita,
    'Agricultural_Use': agri,
    'Industrial_Use': ind,
    'Household_Use': house,
    'Rainfall': rain,
    'Groundwater_Depletion': ground,
    'Scarcity': scarcity_map[scarcity]
}

input_df = pd.DataFrame([input_data])

# Add country columns
for col in columns:
    if col.startswith("Country_"):
        input_df[col] = 0

country_col = f"Country_{country}"
if country_col in input_df.columns:
    input_df[country_col] = 1

# Align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# ================= LAYOUT =================
col1, col2 = st.columns(2)

# -------- INPUT SUMMARY --------
with col1:
    st.subheader("📥 Input Summary")
    st.dataframe(input_df)

# -------- PREDICTION --------
with col2:
    st.subheader("📊 Prediction")

    if st.button("🚀 Predict Water Consumption"):
        result = model.predict(input_df)[0]

        st.metric(
            label="💧 Predicted Consumption",
            value=f"{result:.2f} BCM"
        )

        # Insight message
        if result > 700:
            st.warning("⚠️ High water consumption predicted!")
        elif result > 400:
            st.info("ℹ️ Moderate water consumption.")
        else:
            st.success("✅ Low water consumption.")

# ================= FEATURE IMPORTANCE =================
st.subheader("📊 Feature Importance Analysis")

importances = model.feature_importances_

feat_df = pd.DataFrame({
    'Feature': columns,
    'Importance': importances
})

# -------------------------------
# Combine all Country features
# -------------------------------
country_importance = feat_df[feat_df['Feature'].str.contains("Country")]['Importance'].sum()

# Remove individual country columns
feat_df = feat_df[~feat_df['Feature'].str.contains("Country")]

# Add combined country feature
feat_df = pd.concat([
    feat_df,
    pd.DataFrame({'Feature': ['Country'], 'Importance': [country_importance]})
])

# Sort again
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# -------------------------------
# Plot
# -------------------------------
fig = plt.figure()
plt.barh(feat_df['Feature'], feat_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()

st.pyplot(fig)

# ================= DATA PREVIEW =================
st.subheader("📁 Dataset Preview")

try:
    df = pd.read_csv("data.csv")
    st.dataframe(df.head())
except:
    st.warning("Dataset not found")

# ================= FOOTER =================
st.markdown("---")