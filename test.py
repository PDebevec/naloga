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

data = np.load(open('data.npy', 'rb'), allow_pickle=True)

images = lb.get_image(data)
finding = lb.seperate_image(images, lb.get_label(data[:, 2, 0]))

for img in finding:
    print('img')
    for label in img:
        print('label')
        for imgs in label:
            print(imgs[2])

#print(data)

""" model = SpectralClustering(n_clusters=3)
 res = model.fit(lb.getX(data[:, 4]))

for i,e in enumerate(res.labels_):
    print(e, data[i, 2]) """