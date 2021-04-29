#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train_X = pd.read_csv('X_train.csv', header = None)
train_y = pd.read_csv('y_train.csv', header = None)

print(train_X.shape) #8345个数据 128个features
print(train_y.shape)


# In[67]:


index_dict = {1:[],2:[],3:[],4:[],5:[],6:[]}
for i in range(train_y.shape[0]):
    index_dict[train_y.iloc[i,0]].append(i)
print(index_dict)


# In[81]:


for label in range(1,7):
    index_list = index_dict[label]
    train_part = train_X.iloc[index_list]
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(train_part)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    #print(train_part)
    for k_label in set(labels):
        diff = np.array(train_part.loc[np.array(labels) == k_label,]) -                    - np.array(centers[k_label])

        dist = np.sum(np.square(diff), axis=1)
        UL = dist.mean() + 3*dist.std()
        OutLine = np.where(dist > UL, 1, 0)  #如果是1，就是异常点
        train_part.loc[:, 'outlier'] = OutLine
    rst = train_part.loc[train_part['outlier']== 1]
    print('label is :', label)
    print(list(rst.index.values))
    #res.loc[res['OutLier']==1]    
    #print(train_part)

