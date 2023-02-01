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

data = np.load(open('data.npy', 'rb'), allow_pickle=True)

""" for i in data[:, 4]:
    print(len(i[:][:1400])) """

print(data[:, 4])
