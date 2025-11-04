import joblib

vectorizer = joblib.load("model/vectorizer.pkl")
X_train, y_train, X_test, y_test = joblib.load("model/data_split.pkl")