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


#print(lb.data[:, 4, 0])
#print(lb.get_diff(lb.data[:, 4, 0])[0])

dif = lb.get_diff(lb.data[:, 4, 0])[0]
plt.plot(dif)
plt.plot(lb.data[:, 4, 0][0] / 35)
#print(ss.find_peaks(lb.get_diff(lb.data[:1, 4, 0])[0], height=0, distance=100))
p = ss.find_peaks(lb.get_diff(lb.data[:1, 4, 0])[0], height=0, distance=100)[0]
plt.plot(p, dif[p], 'x', color='black')
plt.show()