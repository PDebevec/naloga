#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.sparse as sparse
from lib import lib as lb
from lib import ml
#import lib2 as lb2

""" graf = 120

d = ss.savgol_filter(lb.data[graf, 4][:300], window_length=20, polyorder=2)

for x in np.where(lb.data[:, 2] == 'Cancer')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='red')
for x in np.where(lb.data[:, 2] == 'Healthy')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='green')
for x in np.where(lb.data[:, 2] == 'Benign')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='blue')
#plt.plot(d)
plt.show() """

#exit()