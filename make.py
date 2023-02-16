import pandas as pd
import dask.dataframe as df
import dask
from lib import ml
import numpy as np
import pickle

file = pd.read_csv('data.csv')

del file['TTP_GS_sig2']
del file['TTP_GS_sig20']
del file['NIR_GS_sig20']
del file['NIR_GS_sig2_G_corr']
del file['NIR_GS_sig20_G_corr']
del file['x_init']
del file['y_init']

npy = file.to_numpy()
npy[:, 3] = ml.to_array(npy[:, 3])
#npy[:, 4] = ml.to_array(npy[:, 4])
#npy[:, 5] = ml.to_array(npy[:, 5])

file = pd.DataFrame(npy, columns=['video', 'ROI', 'finding', 'NIR'])#, 'NIR_corr2', 'NIR_corr20'])

file['video'] = file['video'].astype(int)
file['ROI'] = file['ROI'].astype(int)
file['finding'] = file['finding'].astype(str)

file = file.set_index(['video', 'finding'])

file = file.drop(16091601)
#file = file.drop(16091401)
#file = file.drop(16092101)

file = file.sort_index()

pickle.dump(file, open('fdata.pickle', 'wb'))