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

""" for img in ml.uvideo:
    x = ml.get_x(img, 'NIR_nfp_savg')

    plt.plot(x.T)
    plt.show() """



""" for img in ml.uvideo:
    b = ml.get_x(img, 'NIR_nfp_butter')
    ttpb = ml.get_x(img, 'TTP_butter')
    for bx, ttp, b in zip(b, ttpb, ml.get_l(img, l='finding')):
        c = 'green'
        match b:
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        plt.plot(bx[ttp:], color=c, alpha=0.5)
    plt.title(str(img) + '_butter')
    plt.show() """

""" arr = []
for img in ml.uvideo:
    #plt.title(img)
    X = ml.get_x(img, 'NIR_nfp_butter').T
    ha = []
    i = 0
    a = []
    for x in X:
        x = x.reshape(-1, 1)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(x)
        #acc = ml.get_accuracy(model.labels_, ml.get_l(img, l=2))
        acc = ml.get_accuracy(model.labels_, ml.get_l(img, l=2))
        a.append(acc)
        if acc > 0.85:
            ha.append(i)
        i+=1
    arr.append(ha)
    plt.title(str(img) + '_butter' + '\nmin:' + str(int(np.min(a)*1000)/10) + '% max:' + str(int(np.max(a)*1000)/10) + '% mean:' + str(int(np.mean(a)*1000)/10) + '%')
    plt.plot(a)
    plt.plot(ml.get_diff_indata(X.T))
    #plt.savefig('./graphs/acc&diff/'+str(img)+'.png', dpi=350)
    #plt.clf()
    plt.show() """

#exit()