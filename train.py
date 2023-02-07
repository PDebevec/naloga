import pickle
import numpy as np
from lib import ml
from lib import lib as lb
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

""" x_train,x_test, y_train,y_test = ml.get_xy_data()
x_train = x_train/255
x_test = x_test/255 """

x = np.concatenate(lb.data[:, 4]).ravel().reshape(-1, 1400)/255
#x = np.concatenate(lb.data[:, 4][:750]).ravel().reshape(-1, 1400)/255
y = lb.data[:, 2]

""" model = MiniBatchKMeans(n_clusters=12, batch_size=1024, verbose=1, max_no_improvement=50)

model.fit(x)

pickle.dump(model, open('model.h5', 'wb')) """

model = pickle.load(open('model.h5', 'rb'))

l = ml.separate_labels(model.labels_, y)

for i in range(12):
    print(i, l[0].count(i))
print()

for i in range(12):
    print(i, l[1].count(i))
print()

for i in range(12):
    print(i, l[2].count(i))
print()