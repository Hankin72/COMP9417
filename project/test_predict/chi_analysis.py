import pandas as pd
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import classification_report, confusion_matrix

X_train = pd.read_csv("X_train.csv", header=None)
y_train = pd.read_csv("y_train.csv", header=None).values.ravel()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# print(X_train)
select = SelectKBest(chi2, k=128)
select.fit_transform(X_train, y_train)

score = select.scores_
#print(score)
#print(len(score))

plt.bar(range(len(score)), score)
plt.title("chi2 test score distribution")
plt.ylabel("score")
plt.xlabel("feature code")
plt.show()

scoreList = [(idx, val) for idx, val in enumerate(score)]

scoreList = sorted(scoreList, key=lambda x:x[1], reverse=True)

df = pd.DataFrame()
for idx, score in scoreList:
    df = df.append({'Index':idx, 'chi2-score':score}, ignore_index=True)
df['Index'] = df['Index'].astype(int)
print(df)
df.to_csv("feature choice/chi2.csv", index=False)
