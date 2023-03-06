#import sklearn as sk
import lib as lb
import ml
import sys
import pickle
import time
import tsfel
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it
# nimajo >90% acc na posameznih podatkih 170101 16091301 16093001 16093201
#170108 16091401 16093601 16093801 """16092701"""
#16093201 sranje 16090101 oboje
# za probat namesto unga smth savgol_filter

## vse dat na isti ttp ##
## diff na podatkih za vse videje tu dat v clustering skupi
## np.zeros = decomposition na podatkih, vse videje skupaj v clustering
## algo za misssing values sklearn (sklearn.inpute. ...) ## posamezen video? idk, na vsak roi posebej bi mogli met label
##^https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
## forest za (model_selection)cv hyper parameter
## ## kšna druga funkcija za glajenje savgal in gaussian (morde obe skupi), oz se da tudi svojo (curve_fit)

for img in lb.uvideo: ##največje razlike med podatki jih smooth in morde drops?
    x = lb.get_x(img, 'NIR_nfp_smth')
    ttp = lb.get_x(img, 'TTP_smth')
    dx = lb.get_diff_indata(x)
    #dx = dx[np.argmin(dx[ttp.min():])+ttp.min():] # od ttp naprej

    x = x[:, dx > 0.75]
    
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(x)

    print(img, ml.get_accuracy(model.labels_, lb.get_l(img, l=2)))
    #exit()
#exit()