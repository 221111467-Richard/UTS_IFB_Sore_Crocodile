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



if __name__ == "__main__":
    app.run(debug=True)
