#import sklearn as sk
import lib as lb
import ml
import sys
import pickle
import time
import tsfel
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth_FastICA
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering
#li = pickle.load(open('videolabel.pickle', 'rb'))
#170108 16091401 16093601 16093801 """ 16092701 """

## video 16092001 ni pravilno nfp
## python3 make.py in python3 makef.py
x = lb.get_x(16093601, 'NIR_minmax')
y = lb.get_x(16092001, 'NIR_minmax')

x = np.array(lb.get_nfp(x))
y = np.array(lb.get_nfp(y))

plt.plot(x.T, color='r')
plt.plot(y.T, color='b')

plt.show()

""" cluster = [ 
    MiniBatchKMeans(n_clusters=2)
    ,KMeans(n_clusters=2)
    ,SpectralClustering(n_clusters=2)
    ,Birch(n_clusters=2)
    ,AgglomerativeClustering(n_clusters=2)
]
arr = []
pos = []
for img in lb.uvideo:
    x = lb.get_x(img, 'NIR_nfp')
    arr.append([])
    pos.append([])
    for c in cluster:
        c.fit(x)
        arr[-1].append(ml.get_accuracy(c.labels_, lb.get_l(img)))
    arr[-1] = (arr[-1] == np.max(arr[-1])).astype(int)
    w = arr[-1].mean()
    h = np.where(arr[-1] == 1)[0]+1
    #print(h, arr[-1], np.sum([ 2**y for y in h ]))
    h = np.sum([ 2**y for y in h ])
    pos[-1].append([ h/2, h%2 ])
    #print(w, h)
    #print(arr[-1])
    if lb.videos.loc[img].values[0] == 'Cancer':
        plt.plot(h/2, h%2, 'r+')
    else:
        plt.plot(h/2, h%2, 'bx')
plt.show()

pos = np.array(pos).reshape(-1, 2)
print(pos.shape)

model = AgglomerativeClustering(n_clusters=2)
model.fit(pos)

for l,v in zip(model.labels_, lb.uvideo):
    print(v, l) """

#exit()