#import sklearn as sk
#import lib as lb
import ml
import sys
import pickle
import time
import tsfel
import numpy as np
import pandas as pd
import similaritymeasures as sm
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB ## blo v poslanem (link spodi)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, StackingClassifier #najbulš s pravimi feature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, butter, lfilter, filtfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it
## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8861725/
## https://whimsical.com/classica-video-report-example-XXQEU7ngNqe3sSQLXCMny9
##? from ml impot * ig? zaradi drugih lib
#170108 16091401 16093601 16093801 """16092701"""
## algo za misssing values sklearn
##^https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

for img in ml.uvideo:
    x = ml.get_x(img, 'NIR_nfp_savg')

    plt.plot(x.T)
    plt.show()


## začet pisat poročilo
## ig se mal poigrat z smooth
## doma tudi delat

## pogledat kateri column od 1490 je najbulš >csv
## ne vem doma 


#exit()