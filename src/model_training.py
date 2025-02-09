import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, preprocess_data

def train_model():
    df = load_data(r"C:\Users\white\PycharmProjects\ChurnPrediction\data\customer_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)


    joblib.dump(X_train.columns.tolist(), r"C:\Users\white\PycharmProjects\ChurnPrediction\models\model_features.pkl")

    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    joblib.dump(model, r"C:\Users\white\PycharmProjects\ChurnPrediction\models\churn_model.pkl")
    print("Model and feature names saved successfully!")

if __name__ == "__main__":
    train_model()
