import pickle

from utils import preprocess

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

text = input('review: ')
clean = preprocess(text)
vec = vectorizer.transform([clean])
pred = model.predict(vec)[0]
label = 'positive' if pred == 1 else 'negative'
print(label)
