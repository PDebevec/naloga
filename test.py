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

""" file = pd.read_csv('data.csv')
print(file.info())

file.NIR_GS_sig2 = lb.to_array(file.NIR_GS_sig2)
file.NIR_GS_sig20 = lb.to_array(file.NIR_GS_sig20)
file.NIR_GS_sig2_G_corr = lb.to_array(file.NIR_GS_sig2_G_corr)
file.NIR_GS_sig20_G_corr = lb.to_array(file.NIR_GS_sig20_G_corr)

npy = file.to_numpy()
np.save(open('data.npy', 'wb'), npy)
 """
data = np.load(open('data.npy', 'rb'), allow_pickle=True)

seperate_img = lb.get_image(data)

""" graf = 0

ttp = ss.find_peaks(data[:, 4][graf][:200], distance=150)
print(ttp[0] * 0.2002, data[:, 3][graf])
plt.plot(data[:, 4][graf])
for e in ttp[0]:
    plt.plot(e, data[:, 4][graf][e], 'x')

plt.axvline(int(data[:, 3][graf]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(data[:, 3][graf]/0.2002)) """

fiture, axis = plt.subplots(3,2)
for i,e in enumerate(seperate_img[:3]):
    for k in e[:, 4]:
        axis[i, 0].plot(k)
    for k in e[:, 6]:
        axis[i, 1].plot(k)
plt.show()
