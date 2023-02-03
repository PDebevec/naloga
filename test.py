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
import lib as lb

x_train,x_test, y_train,y_test = lb.get_xy_data()
x_train = x_train/255
x_test = x_test/255

print(x_test.shape, x_train.shape)

#exit()