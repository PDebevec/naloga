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
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import scipy.signal as ss
import scipy.sparse as sparse
import lib as lb
import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth FastICA default AgglomerativeClustering avg:0.8581598572776615 min:0.5365853658536586
#NIR_diff_smth_FastICA

csv = pd.read_csv('all.csv').set_index( ['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster'] )

videos = []
algos = []
for img in pd.unique(lb.data.index.get_level_values(0)):
    x = csv.xs(img, level='img')
    y = np.array(x['acc'].tolist())
    if x.iloc[y.astype(float).argsort()[-1]]['acc'] == 1:
        videos.append(img)
    else:
        algos.append(x.reset_index().iloc[y.astype(float).argsort()[-1]].values)

for img in videos:
    x = csv.xs(img, level='img')
    for algo in algos:
        print(algo)
        print(x.loc[algo[0], algo[1], algo[2], algo[3]]['acc'])
#exit()