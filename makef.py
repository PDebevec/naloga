import lib as lb
import pandas as pd
import pickle

x = pickle.load(open('data1.pickle', 'rb')).drop(170110)
x = x.drop(16092701)
y = pickle.load(open('data2.pickle', 'rb')).drop(16091601)

z = pd.merge(x, y, left_index=True, right_index=True)

z['binary'] = lb.get_binary(z)

z = z.set_index('binary', append=True).sort_index(level=[0,2])
z.to_pickle('data.pickle')

print(z.info())