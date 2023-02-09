#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.sparse as sparse
from lib import lib as lb
from lib import ml
#import lib2 as lb2

""" file = pd.read_csv('data.csv')

del file['TTP_GS_sig2']
del file['TTP_GS_sig20']
del file['NIR_GS_sig20']
del file['NIR_GS_sig2_G_corr']
del file['NIR_GS_sig20_G_corr']
del file['x_init']
del file['y_init']

npy = file.to_numpy()
npy[:, 3] = ml.to_array(npy[:, 3])

file = pd.DataFrame(npy, columns=['video', 'ROI', 'finding', 'NIR'])

file['video'] = file['video'].astype(int)
file['ROI'] = file['ROI'].astype(int)
file['finding'] = file['finding'].astype(str)

file = file.set_index(['video', 'finding', 'ROI'])

model = LocalOutlierFactor(n_neighbors=325)
res = model.fit_predict(np.concatenate(file['NIR'].values).reshape(-1, 1400))
file = file.drop(index=file.iloc[np.where(res == -1)[0]].index)

pickle.dump(file, open('fdata.pickle', 'wb')) """

data = pickle.load(open('fdata.pickle', 'rb'))
data['NIR_minmax'] = lb.get_normalized(data['NIR'])
data['NIR_minmax_img'] = lb.get_normalized_byimg(data['NIR'])
data['NIR_255'] = data['NIR']/255
#print(data.xs('Cancer', level='finding', drop_level=False))

""" l = 'Healthy'
model = LocalOutlierFactor(n_neighbors=int(len(data.xs(l, level='finding')['NIR'])/10))
res = model.fit_predict(np.concatenate(data.xs(l, level='finding')['NIR'].values).reshape(-1, 1400)) """
model = LocalOutlierFactor(n_neighbors=int(len(data)/3.5))
res = model.fit_predict(np.concatenate(data['NIR'].values).reshape(-1, 1400))
data = data.drop(index=data.iloc[np.where(res == -1)[0]].index)
#print(len(np.where(res == -1)[0]))

""" plt.plot(model.negative_outlier_factor_)
plt.show() """

model = AgglomerativeClustering(n_clusters=3*25, linkage='ward')
res = model.fit(np.concatenate(data['NIR'].values).reshape(-1, 1400))

print(res.labels_)
sl = ml.separate_labels(model.labels_, data.index.get_level_values(1))
print(ml.find_biggest_batch(sl, 25))


""" for img in pd.unique(data.index.get_level_values(0)):
    for label in pd.unique(data.loc[img].index.get_level_values(0)):
        c = ''
        match label:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        for x in data.loc[img, label]['NIR_minmax']:
            plt.plot(x, color=c)
    #plt.savefig('graphs/minmax/'+str(img)+'_minmax.png')
    #plt.savefig('graphs/minmax/'+str(img)+'_minmax.png')
    plt.show() """

#exit()