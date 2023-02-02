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
import scipy.sparse as sparse
import lib as lb

model = SpectralClustering(n_clusters=12, assign_labels='cluster_qr', eigen_solver='lobpcg')

res = model.fit(np.concatenate(lb.data[:, 4, 0]).ravel().reshape(1214, 1400)[:1200])

""" print(lb.data[:10, 2, 0])
print(res.labels_[:10]) """

print(lb.separate_labels(res.labels_, lb.data[:1200, 2, 0]))

#print(res.labels_)