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

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global last_df
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file, quotechar='"', sep=',', on_bad_lines='skip')

        base_cols = [
            'customer review', 'review', 'reviews', 'comment', 'comments',
            'text', 'content', 'feedback', 'message', 'messages', 'opinion',
            'body', 'ulasan', 'deskripsi', 'isi', 'caption', 'post', 'tweet',
            'status', 'description', 'response', 'remark', 'testimonial',
            'statement', 'komentar', 'tanggapan', 'pendapat', 'evaluasi',
            'keterangan', 'pesan', 'uraian', 'narasi', 'isi review',
            'isi komentar', 'sentence', 'text tweet'
        ]
        possible_cols = [col for c in base_cols for col in [c, c.capitalize(), c.upper()]]

        review_col = next(
            (col for col in df.columns if col.strip().lower() in [p.lower() for p in possible_cols]),
            None
        )
        if not review_col:
            return render_template("upload.html", error=f"Kolom teks tidak ditemukan. Kolom: {list(df.columns)}")

        df = df[df[review_col].astype(str).str.len() > 5].head(200)
        df['cleaned'] = df[review_col].astype(str).apply(clean_text)
        df['Prediction'] = sentiment_model.predict(vectorizer.transform(df['cleaned']))

        last_df = df  
        results = [{'review': r, 'pred': p} for r, p in zip(df[review_col], df['Prediction'])]
        return render_template("upload.html", results=results)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
