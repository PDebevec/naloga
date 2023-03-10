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
##^^ from ml impot * ig? zaradi drugih lib
#170108 16091401 16093601 16093801 """16092701"""

## np.zeros = decomposition na podatkih, vse videje skupaj v clustering
## algo za misssing values sklearn (sklearn.inpute. ...) ## posamezen video? idk, na vsak roi posebej bi mogli met label
##^https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
## forest v (model_selection)cv hyper parameter



#exit()