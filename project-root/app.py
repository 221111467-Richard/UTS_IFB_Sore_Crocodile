from flask import Flask, render_template, request
import joblib
import re
import io, base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


from wordcloud import WordCloud


app = Flask(__name__)

sentiment_model = joblib.load("model/sentiment_svm.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

negation_words = {
    "tidak", "tak", "tiada", "bukan", "jangan", "belum", "tanpa", "enggak", "gak", "ndak", "nda", "kagak", "ga",
    "ora", "dak", "idak", "ndok", "sing", "endak", "keneh", "ulun", "kada", "tara", "hamo", "takkan",
    "janganlah", "bukannya", "nggak", "nggak usah", "ga usah", "ga mau", "sama sekali tidak",
    "sama sekali bukan", "mana mungkin", "mustahil"
}

filtered_stopwords = [w for w in stopwords if w not in negation_words]
stopwords = set(filtered_stopwords)

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stopwords]
    return " ".join(tokens)


last_df = None
wordcloud_img = None

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    text = request.form['review']
    X = vectorizer.transform([text])
    sentiment = sentiment_model.predict(X)[0]
    return {"review": text, "prediction": sentiment}

if __name__ == "__main__":
    app.run(debug=True)
