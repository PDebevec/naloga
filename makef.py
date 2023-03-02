import lib as lb
import pandas as pd

x = lb.data.drop(170110)
x = x.drop(16092701).reset_index(level=3, drop=True)
y = lb.data2.drop(16091601)

z = pd.merge(x, y, left_index=True, right_index=True)

z.to_pickle('data.pickle')

print(z.info())