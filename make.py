import pandas as pd
import lib as lb
import numpy as np
import pickle

file = pd.read_csv('data.csv')
print(file.info())

file.NIR_GS_sig2 = lb.to_array(file.NIR_GS_sig2)
file.NIR_GS_sig20 = lb.to_array(file.NIR_GS_sig20)
file.NIR_GS_sig2_G_corr = lb.to_array(file.NIR_GS_sig2_G_corr)
file.NIR_GS_sig20_G_corr = lb.to_array(file.NIR_GS_sig20_G_corr)

npy = file.to_numpy()
np.save(open('data.npy', 'wb'), npy)