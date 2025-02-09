from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model_path = r"C:\Users\white\PycharmProjects\ChurnPrediction\models\churn_model.pkl"
model = joblib.load(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])


        df = pd.get_dummies(df, drop_first=True)


        model_features = joblib.load(r"C:\Users\white\PycharmProjects\ChurnPrediction\models\model_features.pkl")
        missing_cols = set(model_features) - set(df.columns)


        for col in missing_cols:
            df[col] = 0


        df = df[model_features]

        prediction = model.predict(df)
        return jsonify({"Churn Prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
