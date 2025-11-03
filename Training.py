import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


df = pd.read_csv("TrainingData.csv")


df = df[['Customer Review', 'Sentiment']]




stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())


negations = {
    "tidak", "tak", "tiada", "bukan", "jangan", "belum", "tanpa",
    "enggak", "gak", "ndak", "nda", "kagak", "ga", "ora", "dak", "idak",
    "ndok", "sing", "endak", "keneh", "ulun kada", "kada", "tara", "hamo",
    "takkan", "janganlah", "bukannya", "nggak usah", "nggak bakal",
    "ga usah", "ga mau", "sama sekali tidak", "sama sekali bukan",
    "mana mungkin", "mustahil"
}

stopwords = stopwords - negations

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def clean_text(text):
    
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = text.split()

    tokens = [stemmer.stem(w) for w in tokens if w not in stopwords]
    return " ".join(tokens)


df['cleaned'] = df['Customer Review'].astype(str).apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X = vectorizer.fit_transform(df['cleaned'])
y = df['Sentiment']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


joblib.dump(vectorizer, "model/vectorizer.pkl")

joblib.dump((X_train, y_train, X_test, y_test), "model/data_split.pkl")

print("✅ Preprocessing selesai. Vectorizer & data split disimpan di folder model/")
