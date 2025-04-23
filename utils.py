import re

import nltk
import pymorphy3
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_ru = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    lemmatized = [morph.normal_forms(word)[0] for word in tokens]
    cleaned = [word for word in lemmatized if word not in stopwords_ru and len(word) > 2]
    return " ".join(cleaned)
