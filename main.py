"""
Symmetric Nonnegative Matrix Factorization of
Elastic-net Regularized Block-wise Weighted Features
for Clustering

Application : ORL, MNIST, CALTECH101, IMAGENET
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
----------------------------------------------------------
------            Main application file              -----

"""

import numpy as np
import os,sys
import argparse

from model import SSNMF_ENBW_C
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# --Clustering and dataset parameters ----------
# ----------------------------------------------
ap.add_argument("-NC",   "--NC", required=True, help="Number of target clusters.")
ap.add_argument("-DA",   "--DA", required=True, help="Dataset: (1) ORL, (2) MNIST-test, (3) CALTECH101-5, (4) IMAGENET.")
ap.add_argument("-N",    "--N", required=True, help="Number of original dataset samples.")
ap.add_argument("-NS",   "--NS", required=True, help="Number of desired dataset samples.")
ap.add_argument("-NCLA", "--NCLA", required=True, help="Number of dataset original classes.")
# --Weighted Sparse Coding parameters ----------
ap.add_argument("-l_l1", "--lambda_l1", required = True, help = "Lambda L1 penalization parameter.")
ap.add_argument("-l_l2", "--lambda_l2", required = True, help = "Lambda L2 penalization parameter.")
ap.add_argument("-beta", "--beta", required = True, help = "Beta penalization parameter for SNMF.")
ap.add_argument("-K","--K", required=True, help="Number of ElasticNet iterations." )
ap.add_argument("-M","--M", required=True, help="Number of blocks for sparse code learning." )
ap.add_argument("-gamma1", "--gamma1", required = True, help = "Gamma1 lower weights bound.")
ap.add_argument("-gamma2", "--gamma2", required = True, help = "Gamma2 upper weights bound.")
ap.add_argument("-tau", "--tau", required = True, help = "Regularization parameter for weights.")
ap.add_argument("-n_w_ite","--number_w_ite", required=True, help="Number of iterations for weights learning.")
# --Optimization parameters --------------------
ap.add_argument("-l_r","--learning_rate", required=True, help="Learning rate for optimization algorithm.")
ap.add_argument("-n_e_ite","--number_external_ite", required=True, help="Number of external iterations.")
ap.add_argument("-n_i_ite","--number_internal_ite", required=True, help="Number of internal iterations.")
# --Other parameters ---------------------------
# ----------------------------------------------
ap.add_argument("-d_s","--display_step", required=True, help="How often to display training information (every display_step times).")
ap.add_argument("-it_sr","--ite_sparse_rep", required=True, help="External iteration at which final sparse representation clustering starts.")
ap.add_argument("-nrows", "--nrows", required=True, help="Number of image rows.")
ap.add_argument("-ncols", "--ncols", required=True, help="Number of image columns.")
ap.add_argument("-in_f", "--input_folder", required = True, help="Input folder.")
ap.add_argument("-ou_f", "--output_folder", required = True, help="Output folder.")
args = vars(ap.parse_args())

NC               = int(args['NC'])
DATASET          = int(args['DA'])
n                = int(args['N'])
ns               = int(args['NS'])
NCLASSES         = int(args['NCLA'])
LambdaL1         = float(args['lambda_l1'])
LambdaL2         = float(args['lambda_l2'])
Beta             = float(args['beta'])
K                = int(args['K'])
M                = int(args['M'])
gamma1           = float(args['gamma1'])
gamma2           = float(args['gamma2'])
tau              = float(args['tau'])
num_w_ite        = int(args['number_w_ite'])
learning_rate    = float(args['learning_rate'])
num_e_ite        = int(args['number_external_ite'])
num_i_ite        = int(args['number_internal_ite'])
display_step     = int(args['display_step'])
ite_sr           = int(args['ite_sparse_rep'])
nrows            = int(args['nrows'])
ncols            = int(args['ncols'])
input_folder     = str(args['input_folder'])
output_folder    = str(args['output_folder'])

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class_model =  SSNMF_ENBW_C( NC, DATASET, \
                 LambdaL1, LambdaL2, Beta, K,M, gamma1, gamma2,tau, num_w_ite,\
                 learning_rate,num_e_ite,num_i_ite, display_step,ite_sr,\
                 input_folder,output_folder,\
                 nrows,ncols, n, ns,NCLASSES)
class_model.train()

