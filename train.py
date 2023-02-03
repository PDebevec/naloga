import pickle
import lib as lb
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

x_train,x_test, y_train,y_test = lb.get_xy_data()
x_train = x_train/255
x_test = x_test/255