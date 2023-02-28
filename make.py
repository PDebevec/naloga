import pandas as pd
import dask
import lib as lb
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer

file = pd.read_csv('data.csv')

del file['TTP_GS_sig2']
del file['TTP_GS_sig20']
del file['NIR_GS_sig20']
del file['NIR_GS_sig2_G_corr']
del file['NIR_GS_sig20_G_corr']
del file['x_init']
del file['y_init']

file['NIR_GS_sig2'] = lb.to_array(file['NIR_GS_sig2'].values)

file = file.rename(columns={'Video':'video', 'ROI_num':'ROI', 'NIR_GS_sig2':'NIR'})

""" npy = file.to_numpy()
npy[:, 3] = lb.to_array(npy[:, 3])
#npy[:, 4] = ml.to_array(npy[:, 4])
#npy[:, 5] = ml.to_array(npy[:, 5])

file = pd.DataFrame(npy, columns=['video', 'ROI', 'finding', 'NIR'])#, 'NIR_corr2', 'NIR_corr20']) """

file['video'] = file['video'].astype(int)
file['ROI'] = file['ROI'].astype(int)
file['finding'] = file['finding'].astype(str)

file = file.set_index(['video', 'finding', 'ROI'])

#file = file.sort_index(level=[0, 2])


file = file.drop(16091601)
#file = file.drop(16091401)
#file = file.drop(16092101)

file['NIR_255'] = file['NIR']/255
file['NIR_minmax'] = lb.get_minmax(file['NIR_255'])
file['NIR_nfp'] = lb.get_nfp(file['NIR_minmax'])
file['NIR_diff'] = lb.get_gaussian_diff(file['NIR_255'], 1)

file['NIR_255_smth'] = lb.get_gaussian(file['NIR_255'].values, 5)
file['NIR_minmax_smth'] = lb.get_gaussian(file['NIR_minmax'].values, 5)
file['NIR_nfp_smth'] = lb.get_gaussian(file['NIR_nfp'].values, 5)
file['NIR_diff_smth'] = lb.get_gaussian(file['NIR_diff'].values, 5)

file['binary'] = lb.get_binary(file)

file.reset_index(inplace=True)
file = file.set_index(['video', 'finding', 'ROI', 'binary'])
file = file.sort_index(level=[0, 2])

pickle.dump(file, open('data.pickle', 'wb'))
print(file.info())

""" import ml

file = ml.decomposition_data('NIR_255_smth', file, 12)
print('1. decomposition')
file = ml.decomposition_data('NIR_minmax_smth', file, 12)
print('2. decomposition')
file = ml.decomposition_data('NIR_diff_smth', file, 12)
print('3. decomposition') """

#pickle.dump(file, open('data.pickle', 'wb'))

""" imglabels = np.hstack((
    pd.unique(lb.data.index.get_level_values(0)).reshape(-1, 1),
    np.array([ pd.unique(lb.data.loc[x].query("finding != 'Healthy'").index.get_level_values(0))[0] for x in pd.unique(lb.data.index.get_level_values(0))] ).reshape(-1, 1)
))

model = LabelBinarizer()
imglabels = np.hstack(( imglabels, model.fit_transform(imglabels[:,1]) ))

df = pd.DataFrame(imglabels, columns=['video', 'label', 'binary'])
df['video'] = df['video'].astype(int)
df['binary'] = df['binary'].astype(int)

df = df.set_index(['video'])
df.to_pickle('videolabel.pickle') """