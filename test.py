#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, MeanShift, DBSCAN, OPTICS, Birch, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
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

model = LocalOutlierFactor(n_neighbors=325)
res = model.fit_predict(np.concatenate(file['NIR'].values).reshape(-1, 1400))
file = file.drop(index=file.iloc[np.where(res == -1)[0]].index)

pickle.dump(file, open('fdata.pickle', 'wb')) """

data = pickle.load(open('fdata.pickle', 'rb'))
data['NIR_minmax'] = lb.get_normalized(data['NIR'])
data['NIR_minmax_img'] = lb.get_normalized_byimg(data['NIR'])
data['NIR_255'] = data['NIR']/255
#print(data.xs('Cancer', level='finding', drop_level=False))

model = LocalOutlierFactor(n_neighbors=int(len(data)/3.5))
res = model.fit_predict(np.concatenate(data['NIR_255'].values).reshape(-1, 1400))
data = data.drop(index=data.iloc[np.where(res == -1)[0]].index)

ml.run_clustering(x=data, img=16092101, time=1400, column='NIR_minmax_img', clusteringfun=KMeans)
ml.run_clustering(x=data, img=16092101, time=1400, column='NIR_minmax_img', clusteringfun=SpectralClustering)
ml.run_clustering(x=data, img=16092101, time=1400, column='NIR_minmax_img', clusteringfun=MiniBatchKMeans)
ml.run_clustering(x=data, img=16092101, time=1400, column='NIR_minmax_img', clusteringfun=AgglomerativeClustering)
ml.run_clustering(x=data, img=16092101, time=1400, column='NIR_minmax_img', clusteringfun=Birch)

#exit()