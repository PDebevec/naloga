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
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.sparse as sparse
from lib import lib as lb
from lib import ml
#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#170108

""" csv = pd.read_csv('arez.csv').set_index(['component_fun', 'setting', 'model', 'video'])
dfcsv = np.array(['component_fun', 'setting',  'model', 'video', 'acc'])
for img in pd.unique(lb.data.index.get_level_values(0)):
    #print(img)
    model = KernelPCA(n_components=12, kernel='cosine')
    x = model.fit_transform(np.concatenate(lb.data.loc[img]['NIR_minmax'].values).reshape(-1 ,1400))
    
    compmodel = type(model).__name__
    
    model = KMeans(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'cosine', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = SpectralClustering(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'cosine', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = MiniBatchKMeans(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'cosine', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'cosine', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = Birch(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'cosine', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))

dfcsv = pd.DataFrame(dfcsv[1:], columns=dfcsv[0])
dfcsv = dfcsv.sort_values(by=['component_fun', 'setting', 'model', 'video']).set_index(['component_fun', 'setting', 'model', 'video'])
csv = pd.concat([csv, dfcsv])
csv.to_csv('arez.csv') """

decomposition = [ FactorAnalysis, FastICA, IncrementalPCA, PCA, TruncatedSVD ]
cluster = [ KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch ]
csv = pd.read_csv('arez.csv').set_index(['component_fun', 'setting', 'model', 'video'])
dfcsv = np.array(['component_fun', 'setting',  'model', 'video', 'acc'])
for img in pd.unique(lb.data.index.get_level_values(0)):
    for d in decomposition:
        model = d(n_components=12)
        x = model.fit_transform(np.concatenate(lb.data.loc[img]['NIR_minmax'].values).reshape(-1 ,1400))
        compmodel = type(model).__name__
        for c in cluster:
            model = c(n_clusters=2)
            model.fit(x)
            dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
dfcsv = pd.DataFrame(dfcsv[1:], columns=dfcsv[0])
dfcsv = dfcsv.sort_values(by=['component_fun', 'setting', 'model', 'video']).set_index(['component_fun', 'setting', 'model', 'video'])
csv = pd.concat([csv, dfcsv])
csv.to_csv('arez.csv')

#exit()