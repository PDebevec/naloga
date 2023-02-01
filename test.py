#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import cv2
import skimage
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import lib as lb
import scipy.sparse as sparse

data = np.load(open('data.npy', 'rb'), allow_pickle=True)

print(data.shape)

print(data[:, 4, 0])

#print(data)

""" model = SpectralClustering(n_clusters=3)
 res = model.fit(lb.getX(data[:, 4]))

for i,e in enumerate(res.labels_):
    print(e, data[i, 2]) """