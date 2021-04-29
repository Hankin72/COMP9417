
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import poisson

import pylab
from sklearn.preprocessing import StandardScaler


# using 3-sigma with each feature column
def delete_outliers(X_csv, y_csv):
    samples, features = X_csv.shape
    del_outliers = []
    for n in range(features):
        data_cut = X_csv[n]
        outliers = []
        #     3 sigma function distribution
        mean = data_cut.mean()
        std = data_cut.std()

        upper = mean + 5 * std
        lower = mean - 3 * std

        for i, m in enumerate(data_cut.index):
            if data_cut[i] < lower or data_cut[i] > upper:
                outliers.append(i)
        del_outliers = del_outliers + outliers
    del_outliers = list(set(del_outliers))
    #  drop the outliers points from original training set: X_train.csv and y_train.csv
    X_train_1 = X_csv.drop(X_csv.index[del_outliers]).reset_index(drop=True)
    y_train_1 = y_csv.drop(y_csv.index[del_outliers]).reset_index(drop=True)

    return X_train_1, y_train_1


# using 3-sigma based on different classes [1,2,3,4,5,6]
def del_discrete_points(X_csv, y_csv):
    deleted_index = []
    samples, features= X_csv.shape
    for p in range(1, 7):
        data_cut = X_csv[y_csv[0 ]==p]
        del_index = []
        for n in range(features):
            # 3 sigma experssion
            mean = data_cut.iloc[:, n].mean()
            std = data_cut.iloc[:, n].std()
            upper = mean + 3 * std
            lower = mean - 3 * std
            for i, m in enumerate(data_cut.index):
                if (data_cut.iloc[i, n] > upper) or (data_cut.iloc[i, n] < lower):
                    del_index.append(m)

        deleted_index = deleted_index + del_index
        deleted_index = list(sorted(set(deleted_index)))
    #  drop the outliers points from original training set: X_train.csv and y_train.csv
    X_train_1 = X_csv.drop(X_csv.index[deleted_index]).reset_index(drop=True)
    y_train_1 = y_csv.drop(y_csv.index[deleted_index]).reset_index(drop=True)
    return X_train_1, y_train_1



