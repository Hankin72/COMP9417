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
from three_sigma import *


# select the feature by computing the correlatiom coefficient among features
def cut_dim_by_2_features_corr(X_tain_clean, y_tain_clean):
    delete_columns_dist = {}
    n = X_tain_clean.shape[1]
    for i in range(n):
        delete_columns_dist[i] = []
        for j in range(i + 1, n):
            corr_i_j = stats.pearsonr(X_tain_clean[i], X_tain_clean[j])
            #  set the threshold 0.96, 9.95, 0.98, ...
            if corr_i_j[0] >= 0.96:
                delete_columns_dist[i].append(j)

    del_columns = []
    for key, values in delete_columns_dist.items():
        del_columns = del_columns + values
    del_columns = list(set(del_columns))
    return del_columns


# select the feature by computing the correlatiom coefficient between features and classes
def cut_dim_by_features_corr_label(X_csv, y_csv, del_1st_dims):
    X_csv = pd.DataFrame(X_csv)
    X_del_dims_1st = X_csv.drop(columns=X_csv.columns[del_1st_dims])
    X_re_fes = pd.DataFrame(X_del_dims_1st)
    #  temp, put, train set and label set together, to analysis
    X_re_fes['label'] = y_csv
    index_corrScore_xy = {}

    # Compute Pearson Correlation between label and each feature,
    for i, true_index in enumerate(X_re_fes.columns):
        if i == len(X_re_fes.columns) - 1:
            break
        a = X_re_fes.iloc[:, i]
        b = X_re_fes.iloc[:, -1]
        corr_ab = stats.pearsonr(a, b)[0]
        corr_ab = round(corr_ab, 7)
        index_corrScore_xy[true_index] = corr_ab
    #     output = pd.DataFrame(
    #         {'True_Index': list(index_corrScore_xy.keys()), 'feature_corr_label': list(index_corrScore_xy.values())})
    #     output.sort_values(by="feature_corr_label", ascending=False, inplace=True)

    del_2nd_index = []
    # set the threshold for feature-label correlation screening
    for key, value in index_corrScore_xy.items():
        if - 0.15 < index_corrScore_xy[key] < 0.15:
            #             del_2nd_index.append(key)
            #         if 0 < index_corrScore_xy[key] < 0.15:
            del_2nd_index.append(key)

    del_2nd_index = list(set(del_2nd_index))
    return del_2nd_index


#  select valid features
def filter_features(X_train_1, y_train_1):
    #  get 1st- del_dim through feature-feature correlations
    del_1st_dims = cut_dim_by_2_features_corr(X_train_1, y_train_1)
    #  get the second-del_dims through feature -label correlations
    del_2nd_dims = cut_dim_by_features_corr_label(X_train_1, y_train_1, del_1st_dims)
    # the whole deleted feature columns index
    del_dims = del_1st_dims + del_2nd_dims
    del_dims = [int(i) for i in del_dims]
    return del_dims


# delete un-valid feature
def del_features(X, y, del_features):
    X = X.drop(columns=X.columns[del_features]).T.reset_index(drop=True).T
    return X, y