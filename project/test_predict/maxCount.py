import numpy as np
import pandas as pd
from collections import defaultdict



network = pd.read_csv("NNetwork.csv")
knn = pd.read_csv("KNN.csv")
rf = pd.read_csv("RandomForest.csv")



df = pd.DataFrame(np.stack([network, knn, rf], axis=1)[:,:,0])


maxCount = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    line = df.iloc[i, :]
    count = defaultdict(int)
    for j in range(line.shape[0]):
        count[line[j]] += 1
    maxList = sorted(list(count.items()), key=lambda x: x[1], reverse=True)
    maxValue = maxList[0][0]
    maxCount[i] = maxValue

df['maxCount'] = maxCount

print(df.iloc[175:200, :])


