import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


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
