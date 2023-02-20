import pandas as pd
import dask
import lib as lb
import ml
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

file['NIR_255'] = file['NIR']/255
file['NIR_minmax'] = lb.get_minmax(file['NIR'])
file['NIR_minmax_img'] = lb.get_minmax_byimg(file['NIR'])
file['NIR_diff'] = lb.get_diff(file['NIR'])

file['NIR_255_smth'] = lb.get_gaussian(file['NIR_255'].values, 15)
file['NIR_minmax_smth'] = lb.get_gaussian(file['NIR_minmax'].values, 15)
file['NIR_minmax_img_smth'] = lb.get_gaussian(file['NIR_minmax_img'].values, 15)
file['NIR_diff_smth'] = lb.get_gaussian(file['NIR_diff'].values, 15)

file = ml.decomposition_data('NIR_255_smth', file, 12)
print('1. decomposition')
file = ml.decomposition_data('NIR_minmax_smth', file, 12)
print('2. decomposition')
file = ml.decomposition_data('NIR_diff_smth', file, 12)
print('3. decomposition')

pickle.dump(file, open('data.pickle', 'wb'))