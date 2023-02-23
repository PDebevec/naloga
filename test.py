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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import scipy.signal as ss
import scipy.sparse as sparse
import lib as lb
import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth_FastICA
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering
#li = pickle.load(open('videolabel.pickle', 'rb'))
#170108 16091401 16092701 16093601 16093801

x = -np.array(lb.data.loc[16091401]['NIR_diff_smth'].values.tolist())

for y in x:
    plt.plot(-y)
    #plt.plot(y)
    plt.plot(-y[:400])
    plt.plot(lb.data.loc[16091401]['NIR_255'].values[0]*np.max(-y))
    w = np.where(-y[:400] < 0+np.max(-y)*0.01)[0]
    plt.plot(w, -y[w], 'x')
    break
plt.show()

#exit()