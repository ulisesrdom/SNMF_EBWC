"""
Symmetric Nonnegative Matrix Factorization of
Elastic-net Regularized Block-wise Weighted Features
for Clustering

Application : ORL, MNIST, CALTECH101, IMAGENET
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
----------------------------------------------------------
------           Auxiliary functions file            -----

"""
import munkres
import numpy as np
import os,sys
from scipy.io import loadmat
from PIL import Image
from time import time
from munkres import Munkres

# Auxiliary functions (data reading) --------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# READ IMAGENET 7341-instance database images ---------------------
# with convolutional features  (5 classes)    ---------------------
# available at https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md
# -----------------------------------------------------------------
def read_prepare_IMAGENET(path,n):
    t0 = time()
    IMAGENET_CONV = loadmat(path+'/ImageNet.mat')
    t1 = time()
    print("Data read in %0.3fs" % (t1 - t0))
    # Reshape data and get n first samples
    data_ind  = np.arange(0,n)
    feat_data            = np.asarray( IMAGENET_CONV['data'][data_ind,0:4096] , dtype=np.float32)
    y_data               = np.asarray( IMAGENET_CONV['data'][data_ind,4096].reshape(n,) , dtype=np.int32)
    print("Data prepare in %0.3fs" % (time() - t1))
    return feat_data,y_data

# READ MNIST test database images ---------------------------------
# -----------------------------------------------------------------
def read_prepare_MNIST_test(path,n):
    t0 = time()
    mnist_raw  = loadmat(path+'/mnist-original.mat')
    mnist      = {'data': mnist_raw['data'].T, \
                  'target': mnist_raw['label'][0],\
                  'COL_NAMES': ['label','data'],\
                  'DESCR': 'mldata.org dataset: mnist-original'}
    t1 = time()
    print("Data read in %0.3fs" % (t1 - t0))
    # Select n from test set
    test_ind   = np.arange(60000,60000+n)
    feat_test, y_test   = mnist['data'][test_ind], mnist['target'][test_ind]
    feat_test           = np.asarray( feat_test , dtype=np.float32)
    y_test              = np.asarray( y_test, dtype=np.int32)
    print("Data prepare in %0.3fs" % (time() - t1))
    return feat_test,y_test

# READ ORL database images      -----------------------------------
# (using order in ORL_FILES.txt -----------------------------------
#  for reproducibility)         -----------------------------------
# -----------------------------------------------------------------
def read_prepare_ORL(path, rs_rows,rs_cols):
    t0         = time()
    c          = 0
    y          = []
    X          = []
    with open('ORL_FILES.txt','r') as txt_file:
       li      = 0
       for line in txt_file:
          # within subject id
          im = Image.open(os.path.join(path , line.split('.')[0]+'.pgm' ))
          im = im.convert("L")
          # resize to given size
          im = im.resize((rs_cols, rs_rows), Image.ANTIALIAS)
          X.append( np.asarray( im, dtype=np.uint8).reshape(-1,) )
          y.append(c)
          li = li + 1
          if li % 10 == 0 :
             c = c+1
    y     = np.asarray(y, dtype=np.int32)
    y_un  = np.unique( y )
    print("Unique labels = {}".format(y_un))
    X     = np.asarray( X, dtype=np.float32 )
    n     = X.shape[0]
    d     = X.shape[1]
    t1    = time()
    print("Data read in %0.3fs" % (t1 - t0))
    return X, y

# READ CALTECH101 1415-instance database images -------------------
# with convolutional features  (5 classes)      -------------------
# available at https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md
# -----------------------------------------------------------------
def read_prepare_CALTECH101(path,n):
    t0 = time()
    CALTECH101_CONV = loadmat(path+'/Caltech101.mat')
    t1 = time()
    print("Data read in %0.3fs" % (t1 - t0))
    # Reshape data and get n first samples
    data_ind  = np.arange(0,n)
    feat_data            = np.asarray( CALTECH101_CONV['data'][data_ind,0:4096] , dtype=np.float32)
    y_data               = np.asarray( CALTECH101_CONV['data'][data_ind,4096].reshape(n,) , dtype=np.int32)
    print("Data prepare in %0.3fs" % (time() - t1))
    return feat_data,y_data

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Subsample small-size groups -------------------------------------
# -----------------------------------------------------------------
def subSample(X,y,d,n,n_s,n_groups):
    # variables for subsample
    X_s          = np.zeros((n_s, d),dtype=np.float32)
    y_s          = np.zeros((n_s,),dtype=np.int32)
    # variables for complement of subsample
    X_comp       = []
    y_comp       = []
    # -----------------------------------------
    GROUP_SIZE_s = int(n_s/n_groups)
    GROUP_SIZE_o = int(n/n_groups)
    print("GROUP_SIZE_O = {}".format(GROUP_SIZE_o))
    print("GROUP_SIZE_S = {}".format(GROUP_SIZE_s))
    ind_orig     = np.arange(n)
    np.random.seed( 0 )
    # initial shuffle to avoid bias on group partition
    np.random.shuffle( ind_orig )
    for gi in range(0,n_groups):
       # subsample per group selecting each time
       # high-variance groups
       if gi == n_groups-1 :
          gi_ind_orig = np.arange( gi*GROUP_SIZE_o, n )
          gi_ind_samp = np.arange( gi*GROUP_SIZE_s, n_s )
       else :
          gi_ind_orig = np.arange( gi*GROUP_SIZE_o, (gi+1)*GROUP_SIZE_o )
          gi_ind_samp = np.arange( gi*GROUP_SIZE_s, (gi+1)*GROUP_SIZE_s )
       CandidateGroupsInd = []
       #CandidateGroupsVar = []
       # identify high-variance group
       #for i in range(0,10):
       gi_samp     = np.random.choice( \
                          ind_orig[ gi_ind_orig ], \
                          size= gi_ind_samp.shape[0], replace=False )
       #gi_var      = np.linalg.norm( np.var( X[ gi_samp,: ] , axis=0 ) )
       CandidateGroupsInd.append( gi_samp )
       #CandidateGroupsVar.append( gi_var )
       # store maximum variance subsamples
       #max_var_s = np.argmax( CandidateGroupsVar )
       #print("--gi = {}, indgroupsize={}".format(gi,CandidateGroupsInd[max_var_s].shape))
       max_var_s   = 0
       X_s[ gi_ind_samp , : ] = X[ CandidateGroupsInd[max_var_s], : ]
       y_s[ gi_ind_samp ] = y[ CandidateGroupsInd[max_var_s] ]
       # store complement of subsamples
       s_complement = np.asarray( list( set(ind_orig[ gi_ind_orig])   -  \
                                        set(CandidateGroupsInd[max_var_s]) ), dtype=np.int32 )
       for i in range(0,s_complement.shape[0]):
          X_comp.append( X[ s_complement[i], : ].reshape(-1) )
          y_comp.append( y[ s_complement[i] ] )
    X_comp = np.asarray(X_comp,dtype=np.float32)
    y_comp = np.asarray(y_comp,dtype=np.int32)
    return X_s, y_s, X_comp, y_comp

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# Code function by Binyuan Hui available at https://github.com/huybery/MvDSCN
# to map cluster labels L2 to given groundtruth labels L1
def mapLabels(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
