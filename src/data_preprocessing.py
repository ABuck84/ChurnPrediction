import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df

def preprocess_data(df):

    df = pd.get_dummies(df, drop_first=True)
    x = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(x, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data(r"C:\Users\white\PycharmProjects\ChurnPrediction\data\customer_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessing done")