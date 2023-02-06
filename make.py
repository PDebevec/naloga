import pandas as pd
import dask.dataframe as df
import dask
from lib import ml
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz

file = pd.read_csv('data.csv')
print(file.info())

""" file.NIR_GS_sig2 = ml.to_array(file.NIR_GS_sig2)
file.NIR_GS_sig20 = ml.to_array(file.NIR_GS_sig20)
file.NIR_GS_sig2_G_corr = ml.to_array(file.NIR_GS_sig2_G_corr)
file.NIR_GS_sig20_G_corr = ml.to_array(file.NIR_GS_sig20_G_corr) """

npy = file.to_numpy()
npy = npy.reshape(1214, 11)

npy[:, 4] = ml.to_array(npy[:, 4])
npy[:, 6] = ml.to_array(npy[:, 6])
npy[:, 7] = ml.to_array(npy[:, 7])
npy[:, 8] = ml.to_array(npy[:, 8])

np.save(open('data.npy', 'wb'), npy)

npy2d = np.concatenate(npy[:, 4]).reshape(-1, 1400)

np.save(open('nir.npy', 'wb'), npy2d)