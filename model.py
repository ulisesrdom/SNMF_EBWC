"""
Symmetric Nonnegative Matrix Factorization of
Elastic-net Regularized Block-wise Weighted Features
for Clustering

Application : ORL, MNIST, CALTECH101, IMAGENET
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
----------------------------------------------------------
------            Model class file              ----------

"""

import numpy as np
import pickle as pk
import sys
from func_tools import (
    read_prepare_ORL,
    read_prepare_MNIST_test,
    read_prepare_CALTECH101,
    read_prepare_IMAGENET,
    subSample,
    mapLabels
)
from sklearn.metrics import normalized_mutual_info_score as NMI_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class SSNMF_ENBW_C(object):
  def __init__(self, \
               NC, DS, \
               Lambda1, Lambda2, Beta, K, M, gamma1, gamma2, tau, num_w_iter,\
               learning_rate,num_ext_iter,num_int_iter, display_step,ite_sr,\
               input_folder,output_folder,\
               nrows,ncols, n,n_s,NCLASSES):
     self.NC        = NC
     self.DS        = DS
     self.Lambda1   = Lambda1
     self.Lambda2   = Lambda2
     self.Beta      = Beta
     self.K         = K
     self.M         = M
     self.gamma1    = gamma1
     self.gamma2    = gamma2
     self.tau       = tau
     self.MAX_W_ITE = num_w_iter
     self.l_r       = learning_rate
     self.num_e_iter= num_ext_iter
     self.num_i_iter= num_int_iter
     self.disp_s    = display_step
     self.num_e_sr  = ite_sr
     self.input_f   = input_folder
     self.output_f  = output_folder
     self.nrows     = nrows
     self.ncols     = ncols
     self.n         = n
     self.n_s       = n_s
     self.NCLASSES  = NCLASSES
     self.d         = 0
     self.load_data()

  def load_data(self):
     # Load images from corresponding database
     # --------------------------------------------------------------------
     print("Obtaining data ...")
     if self.DS == 1 :
        X0, y0      = read_prepare_ORL(self.input_f, self.nrows,self.ncols)
     elif self.DS == 2 :
        X0, y0      = read_prepare_MNIST_test(self.input_f, self.n)
     elif self.DS == 3 :
        X0, y0      = read_prepare_CALTECH101(self.input_f,self.n)
     else :
        X0, y0      = read_prepare_IMAGENET(self.input_f,self.n)
     # Standardize data
     # --------------------------------------------------------------------
     scaler      = StandardScaler()
     scaler.fit( X0 )
     self.X      = scaler.transform( X0 )
     # Normalize data
     # --------------------------------------------------------------------
     scaler2     = MinMaxScaler()
     self.X      = scaler2.fit_transform(self.X)
     self.y      = y0
     self.n      = self.X.shape[0]
     self.d      = self.X.shape[1]
     if self.n_s < self.n :
        self.X_s, self.y_s, self.X_c, self.y_c  = \
                                         subSample(self.X,self.y,\
                                         self.d,self.n,self.n_s,self.NCLASSES)
     else :
        self.X_s = self.X.copy()
        self.y_s = self.y.copy()
        self.X_c = np.asarray([],dtype=np.float32)
        self.y_c = np.asarray([],dtype=np.int32)
     print("DATA = {}".format(self.X))
     print("DATA.shape={}".format(self.X.shape))
     print("LABELS.shape={}".format(self.y.shape))
     print("DATA_S.shape={}".format(self.X_s.shape))
     print("LABELS_S.shape={}".format(self.y_s.shape))
     print("DATA_C.shape={}".format(self.X_c.shape))
     print("LABELS_C.shape={}".format(self.y_c.shape))


  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def getPartialGrad( self, \
                      M, n, block_sizes, X, D, COEF ):
     # Initialize variables
     part_grad   = np.zeros((M,),dtype=np.float32)
     # update individual terms for weights, summing over all samples
     # -------------------------------------------
     for i in range(0,n):
             y_vect  = X[:,i] - np.dot(D,COEF[:,i]) # <-- residual
             inner_i = 0
             for j_w in range(0,M):
                yj_2 = 0
                for i_w in range(inner_i,inner_i + block_sizes[j_w]):
                   yj_2 = yj_2 + (y_vect[i_w]*y_vect[i_w])
                inner_i = inner_i + block_sizes[j_w]
                part_grad[j_w] = part_grad[j_w] + yj_2
     # update individual block weight partial sum involving tau regularization
     for j_w in range(0,M):
             part_grad[j_w] = part_grad[j_w] + self.tau*block_sizes[j_w]
     return part_grad
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def getWeights( self, \
                partial_grad, gamma_inf, gamma_sup, \
                block_sizes, M_block, d, alpha, MAX_ITE ):
     # Initialize variables
     weights = np.zeros((d,d),dtype=np.double)
     sigma   = np.ones((M_block,),dtype=np.double)
     grad    = np.zeros((M_block,),dtype=np.double)
     for i in range(0,M_block):
        grad[i] = sigma[i]*partial_grad[i]
     grad = grad / np.linalg.norm(grad)
     # solve with projected gradient method with eq. + ineq. constraints
     # --------------------------------------------------------------------
     nite    = 0
     for k in range(0,MAX_ITE):
        #alpha = self.backtracking( grad, partial_grad, -grad, sigma, 0.5, 0.9, 1.0, M_block )
        alpha  = 0.01
        sigma  = sigma - alpha*grad
        sigma = np.double( M_block ) * sigma / np.sum(sigma)
        sigma[ sigma < gamma_inf ] = gamma_inf
        sigma[ sigma > gamma_sup ] = gamma_sup
        for i in range(0,M_block):
           grad[i] = sigma[i]*partial_grad[i]
        grad = grad / np.linalg.norm(grad)
        nite          = nite + 1
     sys.stdout.write('WEIGHTS, number of iterations = '+str(nite)+'\n')
     sys.stdout.flush()
     # --------------------------------------------------------------------
     ii = 0
     sum_check = 0.0
     for i in range(0,M_block):
        for k in range(ii, ii+block_sizes[i]):
           weights[k,k]  = sigma[i]
        ii = ii + block_sizes[i]
        sum_check = sum_check + sigma[i]
     sys.stdout.write('(WEIGHTS) -- Sum check = '+str(sum_check)+'\n')
     sys.stdout.flush()
     return weights
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def gradZ_s(self):
     grad_val       = -self.XstWtWXs_ns + np.dot( self.XstWtWXs_ns , self.Z)
     grad_val       = grad_val + self.c1*self.Z + self.L1E_s
     return grad_val
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def gradZ_f(self):
     grad_val       = -self.XstWtWXf_n + np.dot( self.XstWtWXs_n , self.Z_f)
     grad_val       = grad_val + self.c1*self.Z_f + self.L1E_f
     return grad_val
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def gradU(self):
     self.UV_dif    = self.Beta*(self.U-self.V)
     self.UV_t      = np.dot(self.U,self.V.T)
     self.UV_t__X   = self.UV_t - self.Z_hat
     grad_val       = np.dot( self.UV_t__X , self.V ) + self.UV_dif
     return grad_val
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def gradV(self):
     self.UV_dif    = self.Beta*(self.U-self.V)
     self.UV_t      = np.dot(self.U,self.V.T)
     self.UV_t__X   = self.UV_t - self.Z_hat
     grad_val       = np.dot( self.UV_t__X.T , self.U ) - self.UV_dif
     return grad_val

  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def train(self):
     # Start training
     # --------------------------------------------------------------------
     OBJ_VAL         = []
     NORM_DIFF       = []
     NMI_MAX         = []
     NMI_ARI         = []
     ACC             = []
     NMI_MAX_FULL    = []
     NMI_ARI_FULL    = []
     ACC_FULL        = []
     np.random.seed( 0 )
     self.c1         = 2.0*self.Lambda2
     self.L1E_s      = self.Lambda1*np.ones((self.n_s,self.n_s),dtype=np.float32)
     self.Z          = 0.01*np.random.random((self.n_s,self.n_s))
     for i in range(0,self.n_s):
        self.Z[i,i]  = 1.
     self.XstWtWXs_ns= (1./np.double(self.n_s))*np.dot(  self.X_s ,  self.X_s.T )
     print("Solving for ElasticNet Non-negative Coefficients ...")
     for t in range(1,self.K+1):
           gradval       = self.gradZ_s()
           self.Z        = self.Z - 10.*self.l_r*gradval/np.linalg.norm(gradval)
           self.Z[self.Z < 0.] = 0.
     # Solve for block-wise weights
     # with normalized projected gradient descent
     # ---------------------------------------------------------------
     if self.M > 1 :
       print("Solving for block-wise weights ...")
       block_sizes = np.zeros((self.M,),dtype=np.int32)
       for i in range(0,self.M):
          block_sizes[i] = int(self.d/self.M)
       part_grad   = self.getPartialGrad( self.M, self.n_s, block_sizes, \
                                          self.X_s.T, self.X_s.T, self.Z )
       self.W      = self.getWeights(part_grad, self.gamma1, self.gamma2, \
                             block_sizes, self.M, self.d, self.tau, self.MAX_W_ITE)
       print("Weights = {}".format(self.W))
     else :
       self.W      = np.eye(self.d,dtype=np.float32)
     self.XstWtWXs_ns= (1./np.double(self.n_s))*np.dot(  np.dot(self.X_s,np.dot(self.W,self.W)) ,  self.X_s.T )
     self.Z          = 0.01*np.random.random((self.n_s,self.n_s))
     for t in range(1,self.K+1):
           gradval       = self.gradZ_s()
           self.Z        = self.Z - 10.*self.l_r*gradval/np.linalg.norm(gradval)
           self.Z[self.Z < 0.] = 0.
     print("ElasticNet coefficients given block-wise weights solved.")
     print("Forming Similarity Matrix ...")
     self.Z_hat      = np.dot(self.Z.T, self.Z)
     di              = np.sum( self.Z_hat , axis=1 )
     for i in range(0,self.n_s):
      for j in range(0,self.n_s):
         self.Z_hat[i,j] = self.Z_hat[i,j] / (np.sqrt(di[i]) * np.sqrt(di[j]))
     print(self.Z_hat)
     self.U          = np.random.random((self.n_s,self.NC))
     self.V          = self.U.copy()
     self.UV_dif     = self.Beta*(self.U-self.V)
     self.UV_t       = np.dot(self.U,self.V.T)
     self.UV_t__X    = self.UV_t - self.Z_hat
     
     print("Solving SNMF ...")
     for ite in range(1,self.num_e_iter+1):
        print("ITERATION {} -----------------------------".format(ite))
        # Solve for U with projected gradient descent
        # ---------------------------------------------------------------
        print("Solving for U ...")
        for t in range(1,self.num_i_iter+1):
           self.U        = self.U - self.l_r*self.gradU()
           self.U[self.U < 0.]= 0.
        # Solve for V with projected gradient descent
        # ---------------------------------------------------------------
        print("Solving for V ...")
        for t in range(1,self.num_i_iter+1):
           self.V        = self.V - self.l_r*self.gradV()
           self.V[self.V < 0.]= 0.
        # ---------------------------------------------------------------
        #
        if ite % self.disp_s == 0 :
           # Assign cluster labels
           clust_hat = np.argmax( self.V, axis=1)
           if ite >= self.num_e_sr :
              cl_un  = np.unique( clust_hat )
              # Assign cluster labels for all samples based on SR
              # -----------------------------------------------------------
              # Solve for coefficients of all samples
              print("Solving for Coefficients of all samples...")
              np.random.seed( 0 )
              self.Z_f       = 0.01*np.random.random((self.n_s,self.n))
              self.XstWtWXf_n= np.dot(  np.dot(self.X_s,np.dot(self.W,self.W)) ,  self.X.T ) / np.double(self.n)
              self.XstWtWXs_n= np.dot(  np.dot(self.X_s,np.dot(self.W,self.W)) ,  self.X_s.T ) / np.double(self.n)
              self.L1E_f     = self.Lambda1*np.ones((self.n_s,self.n),dtype=np.float32)
              for t in range(1,self.K+1):
                 gradval       = self.gradZ_f()
                 self.Z_f      = self.Z_f - 10.0*self.l_r*gradval/np.linalg.norm(gradval)
                 self.Z_f[self.Z_f < 0.] = 0.
              # Obtain representation error with learned coefficients
              print("Assigning clustering labels ...")
              clust_hat_f    = np.zeros((self.n,),dtype=np.int32)
              for i in range(0,self.n):
                 r_err_l = []
                 for ci in range(0,cl_un.shape[0]):#self.NC):
                    ind_ci = clust_hat==cl_un[ci]
                    r_err_l.append( np.linalg.norm( np.dot(self.W,\
                                       self.X[i,:] - \
                                       np.dot(self.X_s[ind_ci,:].T,self.Z_f[ind_ci,i]) \
                                    ) ) )
                 ind_ci_min     = np.argmin( r_err_l )
                 clust_hat_f[i] = cl_un[ ind_ci_min ]
              # -----------------------------------------------------------
              # Obtain clustering metrics and save results
              # -----------------------------------------------------------
              NMI_ARI_F  = NMI_score(self.y, clust_hat_f, average_method='arithmetic')
              NMI_MAX_F  = NMI_score(self.y, clust_hat_f, average_method='max')
              map_labs_F = mapLabels( self.y, clust_hat_f )
              ACC_F      = accuracy_score( self.y, map_labs_F )
              print("RESULTS ON FULL DATASET -----------------------------------------")
              print("-----------------------------------------------------------------")
              print("Arith. NMI = {}, Max. NMI = {}, Acc = {}".format(NMI_ARI_F, NMI_MAX_F,ACC_F))
              print("-----------------------------------------------------------------")
              print("-----------------------------------------------------------------")
              NMI_MAX_FULL.append( NMI_MAX_F )
              NMI_ARI_FULL.append( NMI_ARI_F )
              ACC_FULL.append( ACC_F )
              pk.dump(NMI_MAX_FULL,open(self.output_f+'/NMI_MAX_FULL.p','wb'))
              pk.dump(NMI_ARI_FULL,open(self.output_f+'/NMI_ARI_FULL.p','wb'))
              pk.dump(ACC_FULL,open(self.output_f+'/ACC_FULL.p','wb'))
              # -----------------------------------------------------------
           #print("clust_hat = {}".format(clust_hat[0:100]))
           #print("true_labs = {}".format(self.y_s[0:100]))
           # Show results for Accuracy and NMI metrics
           NMI_ARI0  = NMI_score(self.y_s, clust_hat, average_method='arithmetic')
           NMI_MAX0  = NMI_score(self.y_s, clust_hat, average_method='max')
           map_labs  = mapLabels( self.y_s, clust_hat )
           ACC0      = accuracy_score( self.y_s, map_labs )
           print("RESULTS ON SUBSAMPLED DATASET --------------")
           print("Arith. NMI = {}, Max. NMI = {}, Acc = {}".format(NMI_ARI0, NMI_MAX0,ACC0))
           l2_UV_err   = np.linalg.norm( self.U - self.V )
           NORM_DIFF.append( l2_UV_err )
           temp      = self.Z_hat - np.dot(self.U, self.V.T)
           obj_v     = (1/(2*np.double(self.n_s)))*np.linalg.norm( temp )**2 + \
                       (1/(2*np.double(self.n_s)))*self.Beta*(l2_UV_err**2)
           print("L2_UV_err = {}, OBJ_val = {}".format(l2_UV_err, obj_v))
           OBJ_VAL.append( obj_v )
           NMI_MAX.append( NMI_MAX0 )
           NMI_ARI.append( NMI_ARI0 )
           ACC.append( ACC0 )
           #pk.dump(self.W,open(self.output_f+'/BW_WEIGHTS1.p','wb'))
           pk.dump(NORM_DIFF,open(self.output_f+'/NORM_DIFF.p','wb'))
           pk.dump(OBJ_VAL,open(self.output_f+'/OBJ_VAL.p','wb'))
           pk.dump(NMI_MAX,open(self.output_f+'/NMI_MAX.p','wb'))
           pk.dump(NMI_ARI,open(self.output_f+'/NMI_ARI.p','wb'))
           pk.dump(ACC,open(self.output_f+'/ACC.p','wb'))
     print("-------------------------------------------------------")

