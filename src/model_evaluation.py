import joblib
import shap
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data

def evaluate_model ():
    df = load_data(r"C:\Users\white\PycharmProjects\ChurnPrediction\data\customer_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model_path = r"C:\Users\white\PycharmProjects\ChurnPrediction\models\churn_model.pkl"
    model = joblib.load(model_path)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    evaluate_model()