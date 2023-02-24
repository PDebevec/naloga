#import sklearn as sk
import numpy as np
import sys
from mpl_toolkits.mplot3d import axes3d
import pickle
import time
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import scipy.sparse as sparse
import lib as lb
import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth_FastICA
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering
#li = pickle.load(open('videolabel.pickle', 'rb'))
#170108 16091401 16092701 16093601 16093801


        
""" for img in pd.unique(lb.data.index.get_level_values(0)):
    nir = np.array(lb.data.loc[img]['NIR_255'].values.tolist())
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(nir[:, ::140])
    print(ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0)), end=' ')
    nfp = np.array(lb.data.loc[img]['NIR_nfp'].values.tolist())
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(nfp[:, ::140])
    print(ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))) """

""" num = 14
for img in pd.unique(lb.data.index.get_level_values(0)):
    arr = []
    for i in range(len(lb.data.loc[img])):
        i1 = 0
        i2 = 0
        x = lb.data.loc[img]['NIR_diff'].values[i]#[:350]
        y = lb.data.loc[img]['NIR_255'].values[i]#[:350]
        x = x/np.max(x)
        y = y/np.max(y)

        for n in np.arange(1, 0, -0.001):
            i = np.where(np.logical_or(x > n, x < -n))[0]
            #print(len(i))
            if len(i) >= num:
                arr.append(i[:num])
                break
    for i in range(len(lb.data.loc[img])):
        x = lb.data.loc[img]['NIR_nfp'].values[i][arr[i]]
        plt.plot(x)
    plt.show()
    plt.plot(np.array(lb.data.loc[img]['NIR_nfp'].values.tolist()).T)
    plt.show() """

""" plt.plot(x/np.max(x))
#plt.plot(i, x[i], 'x')
plt.plot(y/np.max(y))
plt.plot(i1, y[i1]/np.max(y), 'x')
plt.plot(i2, y[i2]/np.max(y), 'x') """
""" plt.plot(y[i])
plt.show() """

#exit()