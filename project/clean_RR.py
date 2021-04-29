import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X_train = np.loadtxt('./MLinTheUnknown-Data/X_train.csv', delimiter=',')
y_train = np.loadtxt('./MLinTheUnknown-Data/y_train.csv', delimiter=',')
X_val = np.loadtxt('./MLinTheUnknown-Data/X_val.csv', delimiter=',')
y_val = np.loadtxt('./MLinTheUnknown-Data/y_val.csv', delimiter=',')

print(X_train.shape)


index=0
for row in X_train:
    if row[1] > 1000  or row[9] > 250 or row[11] > 200 or row[12] > 1000 or row[19] > 1000 or row[20] > 500 or row[23] < -200:
        X_train= np.delete(X_train, (index), axis=0)
        y_train= np.delete(y_train, (index), axis=0)
        index -= 1

    if row[25] > 40 or row[27] > 300  or row[28] > 1500 or row[30] < -100 or row[35] > 40 or row[36] > 300 or row[43] > 40 or row[44] > 300:
        X_train= np.delete(X_train, (index), axis=0)
        y_train= np.delete(y_train, (index), axis=0)
        index -= 1
    if row[65] > 100  or row[95] < -200 or row[99] > 40 or row[100] > 200 or row[107] > 40 or row[108] > 200:
        X_train= np.delete(X_train, (index), axis=0)
        y_train= np.delete(y_train, (index), axis=0)
        index -= 1
    index += 1

print(index)
print(X_train.shape)


# fig, ax = plt.subplots(10,10)
# for i, ax in enumerate(ax.flat):
#     ax.scatter(x=X_train[:,i],y=y_train)
# plt.tight_layout()
# plt.show()


