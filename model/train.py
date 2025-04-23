import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv('../data/interim/labeled_cleaned_data.csv')
df_pos = df[df.label == 1]
df_neg = df[df.label == 0]
df_pos_up = resample(df_pos, replace=True, n_samples=len(df_neg), random_state=42)
df_bal = pd.concat([df_neg, df_pos_up])

X = df_bal.clean_text
y = df_bal.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# saving
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
