#import sklearn as sk
import numpy as np
import sys
from mpl_toolkits.mplot3d import axes3d
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, NMF, MiniBatchNMF, PCA, SparsePCA, TruncatedSVD
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

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
print(X.shape)
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
print(W)
H = model.components_
print(H)

model = MiniBatchNMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
print(W)
H = model.components_
print(H)

#exit()