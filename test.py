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
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering

li = pickle.load(open('videolabel.pickle', 'rb'))
perimg = []
for img in pd.unique(lb.data.index.get_level_values(0)):
    data = lb.data.query("video == "+str(img)).sample(frac=1)#16093801

    model = TruncatedSVD(n_components=2)
    r1 = model.fit_transform(np.array(data['NIR_255'].values.tolist()).T)
    plt.plot(r1)
    model = TruncatedSVD(n_components=2)
    r1 = model.fit_transform(np.array(data['NIR_minmax'].values.tolist()).T)
    plt.plot(r1)
    model = TruncatedSVD(n_components=2)
    r1 = model.fit_transform(np.array(data['NIR_diff'].values.tolist()).T)
    plt.plot(r1)
    plt.show()
    exit()

""" perimg = np.array(perimg)

model = AgglomerativeClustering(n_clusters=2)
model.fit(perimg)

print(ml.get_accuracy(model.labels_, li['label'].values)) """

#plt.show()
#exit()