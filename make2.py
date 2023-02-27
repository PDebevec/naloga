import pandas as pd
import dask
import lib as lb
import numpy as np
import pickle

file = pd.read_csv('data2.csv')

file['x_init'] = [[x] for x in file['x_init']]

del file['x_init']
del file['y_init']

file['Video'] = file['Video'].astype(int)
file['ROI_num'] = file['ROI_num'].astype(int)
file['finding'] = file['finding'].astype(str)

file['timeseries'] = lb.to_array(file['timeseries'].values)
file['timeseries_Gcorr_GS20'] = lb.to_array(file['timeseries_Gcorr_GS20'].values)
file['timeseries_Gcorr_LD_GS20'] = lb.to_array(file['timeseries_Gcorr_LD_GS20'].values)
file['cumul_curve_Gcorr_GS20'] = lb.to_array(file['cumul_curve_Gcorr_GS20'].values)
file['drops_rel'] = lb.to_array(file['drops_rel'].values)
file['drops_rel_med'] = lb.to_array(file['drops_rel_med'].values)
file['cumul_values_at_TTP&drops'] = lb.to_array(file['cumul_values_at_TTP&drops'].values)
file['cumul_intervals_between_TTP&drops'] = lb.to_array(file['cumul_intervals_between_TTP&drops'].values)
file['norm_cumul_values_at_TTP&drops'] = lb.to_array(file['norm_cumul_values_at_TTP&drops'].values)
file['norm_cumul_intervals_between_TTP&drops'] = lb.to_array(file['norm_cumul_intervals_between_TTP&drops'].values)
file = file.rename(columns={'Video':'video', 'ROI_num':'ROI'})
file = file.set_index(['video', 'finding', 'ROI'])
file = file.sort_index(level=[0, 2])

file.to_pickle('data2.pickle')

print(file.info())