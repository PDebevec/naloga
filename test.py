#import sklearn as sk
import numpy as np
import sys
from mpl_toolkits.mplot3d import axes3d
import pickle
import time
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
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import scipy.sparse as sparse
import tsfel
import lib as lb
import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth_FastICA
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering
#li = pickle.load(open('videolabel.pickle', 'rb'))
#170108 16091401 16092701 16093601 16093801

x = lb.get_x(16093601, 'NIR_nfp')
d = lb.get_diff_indata(x.T)[250:]
d = np.where(d == 1)[0][0]+250
y = x[:, d].reshape(-1, 1)

model = AgglomerativeClustering(n_clusters=2)
model.fit(y)
y = model.labels_.reshape(-1, 1)
c0 = np.where(y == 0)[0]
c1 = np.where(y == 1)[0]

#x = np.vstack( (gaussian_filter1d(x[c0].T, sigma=2).T, gaussian_filter1d(x[c1].T, sigma=2).T) ).T
plt.plot(x[c0].T, color='red')
plt.plot(x[c1].T, color='green')
plt.show()


""" a = np.full((1, 1400), 1)
x = np.array(lb.data.loc[170101]['NIR_diff'].values.tolist())[0]
b = np.zeros((1, 1400))
x = np.vstack((a, x, b))
x = np.array(lb.get_minmax(x.T))[:, 1:-1]
y = np.cumsum(x, axis=0)
#plt.plot(x)
plt.plot(y/y.max())
plt.show() """


#exit()