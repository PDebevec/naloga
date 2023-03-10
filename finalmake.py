import pandas as pd
import numpy as np
import pickle
import ml

data1 = pd.read_csv('data.csv')
data2 = pd.read_csv('data2.csv')

data1 = data1.rename(columns={'Video':'video', 'ROI_num':'ROI'})
data2 = data2.rename(columns={'Video':'video', 'ROI_num':'ROI'})

data1 = data1.set_index()
data2 = data2.set_index()

