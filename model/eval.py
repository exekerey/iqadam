import pickle

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

matplotlib.use('TkAgg')  # mac stuff
df = pd.read_csv('../data/interim/labeled_cleaned_data.csv')
df_pos = df[df.label == 1]
df_neg = df[df.label == 0]
df_pos_up = resample(df_pos, replace=True, n_samples=len(df_neg), random_state=42)
df_bal = pd.concat([df_neg, df_pos_up])

X = df_bal.clean_text
y = df_bal.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred, zero_division=1))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.xticks([0, 1], ['neg', 'pos'])
plt.yticks([0, 1], ['neg', 'pos'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xlabel('pred')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()
