import pandas as pd
import dask.dataframe as df
import dask
import lib as lb
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz

file = pd.read_csv('data.csv')
print(file.info())

""" file.NIR_GS_sig2 = lb.to_array(file.NIR_GS_sig2)
file.NIR_GS_sig20 = lb.to_array(file.NIR_GS_sig20)
file.NIR_GS_sig2_G_corr = lb.to_array(file.NIR_GS_sig2_G_corr)
file.NIR_GS_sig20_G_corr = lb.to_array(file.NIR_GS_sig20_G_corr) """

npy = file.to_numpy()
npy = npy.reshape(1214, -1, 1)
npy[:, 4, 0] = lb.to_array(npy[:, 4, 0])
npy[:, 6, 0] = lb.to_array(npy[:, 6, 0])
npy[:, 7, 0] = lb.to_array(npy[:, 7, 0])
npy[:, 8, 0] = lb.to_array(npy[:, 8, 0])

np.save(open('data.npy', 'wb'), npy)
