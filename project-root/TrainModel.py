import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

vectorizer = joblib.load("model/vectorizer.pkl")
X_train, y_train, X_test, y_test = joblib.load("model/data_split.pkl")


model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("🔎 Evaluasi Model SVM")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, "model/sentiment_svm.pkl")
print("✅ Model SVM berhasil dilatih & disimpan di model/sentiment_svm.pkl")