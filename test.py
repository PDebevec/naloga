#import sklearn as sk
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.sparse as sparse
from scipy.ndimage import gaussian_filter
from lib import lib as lb
from lib import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#16091401 170108

x_train = lb.data.xs('Cancer', level=1, drop_level=False)['NIR_255'].sample(frac=1)
x_test, x_train = x_train[200:], x_train[:200]
temp = lb.data.xs('Benign', level=1, drop_level=False)['NIR_255'].sample(frac=1)
x_test, x_train = pd.concat([ x_test, temp[200:] ]), pd.concat([ x_train, temp[:200] ])
temp = lb.data.xs('Healthy', level=1, drop_level=False)['NIR_255'].sample(frac=1)
x_test, x_train = pd.concat([ x_test, temp[200:] ]).to_frame().sample(frac=1), pd.concat([ x_train, temp[:200] ]).to_frame().sample(frac=1)


model = KNeighborsClassifier()
model.fit(np.concatenate(x_train['NIR_255'].values).reshape(-1, 1400), x_train.index.get_level_values(1))

res = model.predict(np.concatenate(x_test['NIR_255'].values).reshape(-1, 1400))

count = 0
for i in range(len(res)):
    if res[i] == x_test.index.get_level_values(1)[i]:
        count+=1
print(len(res), count, (count / len(res)))

model = LocalOutlierFactor()
model.fit(np.concatenate(x_train['NIR_255'].values).reshape(-1, 1400), x_train.index.get_level_values(1))

res = model.predict(np.concatenate(x_test['NIR_255'].values).reshape(-1, 1400))

count = 0
for i in range(len(res)):
    if res[i] == x_test.index.get_level_values(1)[i]:
        count+=1
print(len(res), count, (count / len(res)))

#exit()