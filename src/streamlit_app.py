import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_path = r"C:\Users\white\PycharmProjects\ChurnPrediction\models\churn_model.pkl"
feature_path = r"C:\Users\white\PycharmProjects\ChurnPrediction\models\model_features.pkl"

model = joblib.load(model_path)
model_features = joblib.load(feature_path)

data_path = r"C:\Users\white\PycharmProjects\ChurnPrediction\data\customer_data.csv"
df = pd.read_csv(data_path)

st.title("Customer Churn Prediction System")

st.sidebar.header("Customer Details")

monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, step=1.0)
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=100, step=1)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
input_data = pd.DataFrame([[monthly_charges, tenure, contract]], columns=["MonthlyCharges", "Tenure", "Contract"])
input_data = pd.get_dummies(input_data, drop_first=True)

missing_cols = set(model_features) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

input_data = input_data[model_features]

if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_result = "Likely to Churn" if prediction[0] == 1 else "Not Likely to Churn"

    st.subheader("Prediction Result:")
    st.write(f"**Customer is {churn_result}**")

    st.write(f"Raw Model Output: {int(prediction[0])}")



st.subheader("Data Visualizations")

if st.sidebar.checkbox("Show Churn Distribution"):
    fig, ax = plt.subplots()
    sns.barplot(x=df["Churn"].value_counts().index, y=df["Churn"].value_counts().values, ax=ax)
    st.pyplot(fig)

if st.sidebar.checkbox("Show Service Usage Trends"):
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Tenure", y="MonthlyCharges", hue="Churn", ax=ax)
    st.pyplot(fig)

if st.sidebar.checkbox("Show Demographic Influence"):
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Contract"], y=df["MonthlyCharges"], hue=df["Churn"], ax=ax)
    st.pyplot(fig)