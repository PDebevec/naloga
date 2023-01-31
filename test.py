#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import cv2
import skimage
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy

file = pd.read_csv('data.csv')
print(file.info())

sig2 = pd.DataFrame(file, columns=['TTP_GS_sig2', 'NIR_GS_sig2'])
sig20 = pd.DataFrame(file, columns=['TTP_GS_sig20', 'NIR_GS_sig20'])
G_corr = pd.DataFrame(file, columns=['NIR_GS_sig2_G_corr', 'NIR_GS_sig20_G_corr'])

nir = pd.DataFrame(sig2, columns=['NIR_GS_sig2']).to_numpy()
for i, e in enumerate(nir):
    arr = []
    arr.append(np.fromstring(e[1:-1], sep=','))
    print(np.fromstring(e[1:-1], sep=','))
    #print(arr[0])





""" video = pd.DataFrame(file, columns=['Video']).to_numpy()
finding = pd.DataFrame(file, columns=['finding']).to_numpy()
sig2 = pd.DataFrame(file, columns=['TTP_GS_sig2', 'NIR_GS_sig2']).to_numpy()
sig20 = pd.DataFrame(file, columns=['TTP_GS_sig20', 'NIR_GS_sig20']).to_numpy()
G_corr = pd.DataFrame(file, columns=['NIR_GS_sig2_G_corr', 'NIR_GS_sig20_G_corr']).to_numpy()
xy_init = pd.DataFrame(file, columns=['x_init', 'y_init']).to_numpy() """

#print(file.info())