#import sklearn as sk
#import lib as lb
#import ml
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it
import ml

#ml.test_decomposition()
ml.test_cluster()

""" def get_minmax(X):
    arr = []
    for x in X:
        mx = np.max(x)
        mn = np.min(x)
        arr.append( np.array((x - mn) / (mx - mn)) )
    return arr

file = pickle.load(open('data.pickle', 'rb'))
file['NIR_255'] = file['NIR']/255
file['NIR_minmax'] = get_minmax(file['NIR_255'])

file.to_pickle('data.pickle') """

#file = pickle.load(open('data.pickle', 'rb'))

""" file['NIR_255_smth'] = lb.get_gaussian(file['NIR_255'].values, 20)
file['NIR_nfp_smth'] = lb.get_nfp(file['NIR_255_smth'].values)
file['drops'], file['drops_mean'] = lb.get_drop_mean(file['NIR_nfp_smth'])

def get_accuracy(labels_, y):
    model = LabelBinarizer()
    res = model.fit_transform(y).reshape(-1)
    acc = accuracy_score(res, labels_)
    return max([ acc, (acc-1)*-1 ])

def clustering_on_column(column):
    open('./csv/temp/'+column+'.csv', 'w').close()

    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2, n_init=3)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2, threshold=0.05)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    f = open('./csv/temp/'+column+'.csv', 'a')
    f.write('video,clustering,acc,time\n')
    for img in pd.unique(file.index.get_level_values(0)):
        x = file.loc[img][column]
        for c in cluster:
            st = time.time()
            c.fit(np.array(x.values.tolist()))
            et = time.time() - st
            #print(column, img, type(c).__name__, get_accuracy(c.labels_, x.index.get_level_values(0)), et*1000, sep=',')
            f.write(str(img)+','+type(c).__name__+','+str(get_accuracy(c.labels_, x.index.get_level_values(0)))+','+str(et*1000)+'\n')
    return """

#import lib2 as lb2
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
#NIR_diff_smth_FastICA
#NIR_diff, FactorAnalysis, TruncatedSVD, AgglomerativeClustering
#li = pickle.load(open('videolabel.pickle', 'rb'))
#170108 16091401 16093601 16093801 """16092701"""
#16093201 sranje 16090101 oboje
# za probat namesto unga smth savgol_filter
""" model = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain(),
        np.array(file['NIR_minmax'].values.tolist()),
        verbose=1)

model.to_pickle('tsd.pickle') """

""" tsd = pickle.load(open('tsd.pickle', 'rb'))
sc = [
'0_Absolute energy',
'0_Area under the curve',
'0_Autocorrelation',
'0_Centroid',
'0_ECDF Percentile Count_0',
'0_ECDF Percentile Count_1',
'0_ECDF Percentile_0',
'0_ECDF Percentile_1',
'0_ECDF_0',
'0_ECDF_1',
'0_ECDF_2',
'0_ECDF_3',
'0_ECDF_4',
'0_ECDF_5',
'0_ECDF_6',
'0_ECDF_7',
'0_ECDF_8',
'0_ECDF_9',
'0_Entropy',
'0_FFT mean coefficient_1',
'0_FFT mean coefficient_2',
'0_FFT mean coefficient_3',
'0_FFT mean coefficient_4',
'0_FFT mean coefficient_5',
'0_FFT mean coefficient_6',
'0_FFT mean coefficient_7',
'0_FFT mean coefficient_8',
'0_FFT mean coefficient_9',
'0_FFT mean coefficient_10',
'0_FFT mean coefficient_11',
'0_FFT mean coefficient_12',
'0_FFT mean coefficient_13',
'0_FFT mean coefficient_14',
'0_FFT mean coefficient_15',
'0_FFT mean coefficient_16',
'0_FFT mean coefficient_17',
'0_FFT mean coefficient_18',
'0_FFT mean coefficient_19',
'0_FFT mean coefficient_20',
'0_FFT mean coefficient_21',
'0_FFT mean coefficient_22',
'0_FFT mean coefficient_23',
'0_FFT mean coefficient_24',
'0_FFT mean coefficient_25',
'0_FFT mean coefficient_26',
'0_FFT mean coefficient_27',
'0_FFT mean coefficient_28',
'0_FFT mean coefficient_29',
'0_FFT mean coefficient_30',
'0_FFT mean coefficient_31',
'0_FFT mean coefficient_32',
'0_FFT mean coefficient_33',
'0_FFT mean coefficient_34',
'0_FFT mean coefficient_35',
'0_FFT mean coefficient_36',
'0_FFT mean coefficient_37',
'0_FFT mean coefficient_38',
'0_FFT mean coefficient_39',
'0_FFT mean coefficient_40',
'0_FFT mean coefficient_41',
'0_FFT mean coefficient_42',
'0_FFT mean coefficient_43',
'0_FFT mean coefficient_44',
'0_FFT mean coefficient_45',
'0_FFT mean coefficient_46',
'0_FFT mean coefficient_47',
'0_FFT mean coefficient_48',
'0_FFT mean coefficient_49',
'0_FFT mean coefficient_50',
'0_FFT mean coefficient_51',
'0_FFT mean coefficient_52',
'0_FFT mean coefficient_53',
'0_FFT mean coefficient_54',
'0_FFT mean coefficient_55',
'0_FFT mean coefficient_56',
'0_FFT mean coefficient_57',
'0_FFT mean coefficient_58',
'0_FFT mean coefficient_59',
'0_FFT mean coefficient_60',
'0_FFT mean coefficient_61',
'0_FFT mean coefficient_62',
'0_FFT mean coefficient_63',
'0_FFT mean coefficient_64',
'0_FFT mean coefficient_65',
'0_FFT mean coefficient_66',
'0_FFT mean coefficient_67',
'0_FFT mean coefficient_68',
'0_FFT mean coefficient_69',
'0_FFT mean coefficient_70',
'0_FFT mean coefficient_71',
'0_FFT mean coefficient_72',
'0_FFT mean coefficient_73',
'0_FFT mean coefficient_74',
'0_FFT mean coefficient_75',
'0_FFT mean coefficient_76',
'0_FFT mean coefficient_77',
'0_FFT mean coefficient_78',
'0_FFT mean coefficient_79',
'0_FFT mean coefficient_80',
'0_FFT mean coefficient_81',
'0_FFT mean coefficient_82',
'0_FFT mean coefficient_83',
'0_FFT mean coefficient_84',
'0_FFT mean coefficient_85',
'0_FFT mean coefficient_86',
'0_FFT mean coefficient_87',
'0_FFT mean coefficient_88',
'0_FFT mean coefficient_89',
'0_FFT mean coefficient_90',
'0_FFT mean coefficient_91',
'0_FFT mean coefficient_92',
'0_FFT mean coefficient_93',
'0_FFT mean coefficient_94',
'0_FFT mean coefficient_95',
'0_FFT mean coefficient_96',
'0_FFT mean coefficient_97',
'0_FFT mean coefficient_98',
'0_FFT mean coefficient_99',
'0_FFT mean coefficient_100',
'0_FFT mean coefficient_101',
'0_FFT mean coefficient_102',
'0_FFT mean coefficient_103',
'0_FFT mean coefficient_104',
'0_FFT mean coefficient_105',
'0_FFT mean coefficient_106',
'0_FFT mean coefficient_107',
'0_FFT mean coefficient_108',
'0_FFT mean coefficient_109',
'0_FFT mean coefficient_110',
'0_FFT mean coefficient_111',
'0_FFT mean coefficient_112',
'0_FFT mean coefficient_113',
'0_FFT mean coefficient_114',
'0_FFT mean coefficient_115',
'0_FFT mean coefficient_116',
'0_FFT mean coefficient_117',
'0_FFT mean coefficient_118',
'0_FFT mean coefficient_119',
'0_FFT mean coefficient_120',
'0_FFT mean coefficient_121',
'0_FFT mean coefficient_122',
'0_FFT mean coefficient_123',
'0_FFT mean coefficient_124',
'0_FFT mean coefficient_125',
'0_FFT mean coefficient_126',
'0_FFT mean coefficient_127',
'0_FFT mean coefficient_128',
'0_FFT mean coefficient_129',
'0_FFT mean coefficient_130',
'0_FFT mean coefficient_131',
'0_FFT mean coefficient_132',
'0_FFT mean coefficient_133',
'0_FFT mean coefficient_134',
'0_FFT mean coefficient_135',
'0_FFT mean coefficient_136',
'0_FFT mean coefficient_137',
'0_FFT mean coefficient_138',
'0_FFT mean coefficient_139',
'0_FFT mean coefficient_140',
'0_FFT mean coefficient_141',
'0_FFT mean coefficient_142',
'0_FFT mean coefficient_143',
'0_FFT mean coefficient_144',
'0_FFT mean coefficient_145',
'0_FFT mean coefficient_146',
'0_FFT mean coefficient_147',
'0_FFT mean coefficient_148',
'0_FFT mean coefficient_149',
'0_FFT mean coefficient_150',
'0_FFT mean coefficient_151',
'0_FFT mean coefficient_152',
'0_FFT mean coefficient_153',
'0_FFT mean coefficient_154',
'0_FFT mean coefficient_155',
'0_FFT mean coefficient_156',
'0_FFT mean coefficient_157',
'0_FFT mean coefficient_158',
'0_FFT mean coefficient_159',
'0_FFT mean coefficient_160',
'0_FFT mean coefficient_161',
'0_FFT mean coefficient_162',
'0_FFT mean coefficient_163',
'0_FFT mean coefficient_164',
'0_FFT mean coefficient_165',
'0_FFT mean coefficient_166',
'0_FFT mean coefficient_167',
'0_FFT mean coefficient_168',
'0_FFT mean coefficient_169',
'0_FFT mean coefficient_170',
'0_FFT mean coefficient_171',
'0_FFT mean coefficient_172',
'0_FFT mean coefficient_173',
'0_FFT mean coefficient_174',
'0_FFT mean coefficient_175',
'0_FFT mean coefficient_176',
'0_FFT mean coefficient_177',
'0_FFT mean coefficient_178',
'0_FFT mean coefficient_179',
'0_FFT mean coefficient_180',
'0_FFT mean coefficient_181',
'0_FFT mean coefficient_182',
'0_FFT mean coefficient_183',
'0_FFT mean coefficient_184',
'0_FFT mean coefficient_185',
'0_FFT mean coefficient_186',
'0_FFT mean coefficient_187',
'0_FFT mean coefficient_188',
'0_FFT mean coefficient_189',
'0_FFT mean coefficient_190',
'0_FFT mean coefficient_191',
'0_FFT mean coefficient_192',
'0_FFT mean coefficient_193',
'0_FFT mean coefficient_194',
'0_FFT mean coefficient_195',
'0_FFT mean coefficient_196',
'0_FFT mean coefficient_197',
'0_FFT mean coefficient_198',
'0_FFT mean coefficient_199',
'0_FFT mean coefficient_200',
'0_FFT mean coefficient_201',
'0_FFT mean coefficient_202',
'0_FFT mean coefficient_203',
'0_FFT mean coefficient_204',
'0_FFT mean coefficient_205',
'0_FFT mean coefficient_206',
'0_FFT mean coefficient_207',
'0_FFT mean coefficient_208',
'0_FFT mean coefficient_209',
'0_FFT mean coefficient_210',
'0_FFT mean coefficient_211',
'0_FFT mean coefficient_212',
'0_FFT mean coefficient_213',
'0_FFT mean coefficient_214',
'0_FFT mean coefficient_215',
'0_FFT mean coefficient_216',
'0_FFT mean coefficient_217',
'0_FFT mean coefficient_218',
'0_FFT mean coefficient_219',
'0_FFT mean coefficient_220',
'0_FFT mean coefficient_221',
'0_FFT mean coefficient_222',
'0_FFT mean coefficient_223',
'0_FFT mean coefficient_224',
'0_FFT mean coefficient_225',
'0_FFT mean coefficient_226',
'0_FFT mean coefficient_227',
'0_FFT mean coefficient_228',
'0_FFT mean coefficient_229',
'0_FFT mean coefficient_230',
'0_FFT mean coefficient_231',
'0_FFT mean coefficient_232',
'0_FFT mean coefficient_233',
'0_FFT mean coefficient_234',
'0_FFT mean coefficient_235',
'0_FFT mean coefficient_236',
'0_FFT mean coefficient_237',
'0_FFT mean coefficient_238',
'0_FFT mean coefficient_239',
'0_FFT mean coefficient_240',
'0_FFT mean coefficient_241',
'0_FFT mean coefficient_242',
'0_FFT mean coefficient_243',
'0_FFT mean coefficient_244',
'0_FFT mean coefficient_245',
'0_FFT mean coefficient_246',
'0_FFT mean coefficient_247',
'0_FFT mean coefficient_248',
'0_FFT mean coefficient_249',
'0_FFT mean coefficient_250',
'0_FFT mean coefficient_251',
'0_FFT mean coefficient_252',
'0_FFT mean coefficient_253',
'0_FFT mean coefficient_254',
'0_FFT mean coefficient_255',
'0_FFT mean coefficient_',
'0_Fundamental frequency',
'0_Histogram_0',
'0_Histogram_1',
'0_Histogram_2',
'0_Histogram_3',
'0_Histogram_4',
'0_Histogram_5',
'0_Histogram_6',
'0_Histogram_7',
'0_Histogram_8',
'0_Histogram_9',
'0_Human range energy',
'0_Interquartile range',
'0_Kurtosis',
'0_LPCC_0',
'0_LPCC_1',
'0_LPCC_2',
'0_LPCC_3',
'0_LPCC_4',
'0_LPCC_5',
'0_LPCC_6',
'0_LPCC_7',
'0_LPCC_8',
'0_LPCC_9',
'0_LPCC_10',
'0_LPCC_11',
'0_MFCC_0',
'0_MFCC_1',
'0_MFCC_2',
'0_MFCC_3',
'0_MFCC_4',
'0_MFCC_5',
'0_MFCC_6',
'0_MFCC_7',
'0_MFCC_8',
'0_MFCC_9',
'0_MFCC_10',
'0_MFCC_11',
'0_Max',
'0_Max power spectrum',
'0_Maximum frequency',
'0_Mean',
'0_Mean absolute deviation',
'0_Mean absolute diff',
'0_Mean diff',
'0_Median',
'0_Median absolute deviation',
'0_Median absolute diff',
'0_Median diff',
'0_Median frequency',
'0_Min',
'0_Negative turning points',
'0_Neighbourhood peaks',
'0_Peak to peak distance',
'0_Positive turning points',
'0_Power bandwidth',
'0_Root mean square',
'0_Signal distance',
'0_Skewness',
'0_Slope',
'0_Spectral centroid',
'0_Spectral decrease',
'0_Spectral distance',
'0_Spectral entropy',
'0_Spectral kurtosis',
'0_Spectral positive turning points',
'0_Spectral roll-off',
'0_Spectral roll-on',
'0_Spectral skewness',
'0_Spectral slope',
'0_Spectral spread',
'0_Spectral variation',
'0_Standard deviation',
'0_Sum absolute diff',
'0_Total energy',
'0_Variance',
'0_Wavelet absolute mean_0',
'0_Wavelet absolute mean_1',
'0_Wavelet absolute mean_2',
'0_Wavelet absolute mean_3',
'0_Wavelet absolute mean_4',
'0_Wavelet absolute mean_5',
'0_Wavelet absolute mean_6',
'0_Wavelet absolute mean_7',
'0_Wavelet absolute mean_8',
'0_Wavelet energy_0',
'0_Wavelet energy_1',
'0_Wavelet energy_2',
'0_Wavelet energy_3',
'0_Wavelet energy_4',
'0_Wavelet energy_5',
'0_Wavelet energy_6',
'0_Wavelet energy_7',
'0_Wavelet energy_8',
'0_Wavelet entropy',
'0_Wavelet standard deviation_0',
'0_Wavelet standard deviation_1',
'0_Wavelet standard deviation_2',
'0_Wavelet standard deviation_3',
'0_Wavelet standard deviation_4',
'0_Wavelet standard deviation_5',
'0_Wavelet standard deviation_6',
'0_Wavelet standard deviation_7',
'0_Wavelet standard deviation_8',
'0_Wavelet variance_0',
'0_Wavelet variance_1',
'0_Wavelet variance_2',
'0_Wavelet variance_3',
'0_Wavelet variance_4',
'0_Wavelet variance_5',
'0_Wavelet variance_6',
'0_Wavelet variance_7',
'0_Wavelet variance_8',
'0_Zero crossing rate'
]
tsd = tsd.reindex(sc, axis=1)

tsd = pd.DataFrame(index=file.index, columns=tsd.columns, data=tsd.values)

for col in tsd.columns:
    print(col)

tsd.to_pickle('tsd.pickle')

print(tsd.info()) """

#tsd.sort_index()

#clustering_on_column('drops_mean')

#print(file.info())
#exit()