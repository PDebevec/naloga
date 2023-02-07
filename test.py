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

""" file = pd.read_csv('data.csv')

del file['TTP_GS_sig2']
del file['TTP_GS_sig20']
del file['NIR_GS_sig20']
del file['NIR_GS_sig2_G_corr']
del file['NIR_GS_sig20_G_corr']
del file['x_init']
del file['y_init']

npy = file.to_numpy()
npy[:, 3] = ml.to_array(npy[:, 3])

file = pd.DataFrame(npy, columns=['video', 'ROI', 'finding', 'NIR'])

file['video'] = file['video'].astype(int)
file['ROI'] = file['ROI'].astype(int)
file['finding'] = file['finding'].astype(str)

file = file.set_index(['video', 'finding', 'ROI'])

pickle.dump(file, open('fdata.pickle', 'wb')) """

data = pickle.load(open('fdata.pickle', 'rb'))
data['NIR_minmax'] = lb.get_normalized(data['NIR'])

#print(data.xs('Cancer', level='finding', drop_level=False))
print(data.loc[slice(None), 'Cancer'])

#exit()