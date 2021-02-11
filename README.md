# SNMF_EBWC
Python implementation of Symmetric Nonnegative Matrix Factorization of Elastic-net Regularized Block-wise Weighted Features for Clustering.

## Python libraries required
 * munkres (for mapping cluster labels to groundtruth labels) [https://pypi.org/project/munkres/]
 * numpy
 * pickle (only to save results in a pickle file during learning)
 * PIL (to read ORL input images)
 * scipy (to read MNIST, CALTECH101 and IMAGENET data)

## Usage
* To reproduce ORL database results:
`python main.py -NC 40 -DA 1 -N 400 -NS 400 -NCLA 40 -l_l1 0.01  -l_l2 0.1   -beta 0.01   -K 2000 -M 15  -gamma1 0.5 -gamma2 1.25 -tau 0.00001 -n_w_ite 2000 -l_r 0.05 -n_e_ite 50 -n_i_ite 2000 -d_s 1 -it_sr 51 -nrows 72 -ncols 59 -in_f ORL_FOLDER -ou_f OUTPUT_FOLDER`
<br>

* To reproduce MNIST (3 classes) test database results:
`python main.py -NC 3 -DA 2 -N 3147 -NS 400 -NCLA 3 -l_l1 0.01  -l_l2 0.01  -beta 1.0    -K 2000 -M 15  -gamma1 0.5 -gamma2 1.25 -tau 0.00001 -n_w_ite 2000 -l_r 0.005 -n_e_ite 40 -n_i_ite 2000 -d_s 1 -it_sr 40 -nrows 28 -ncols 28 -in_f MNIST_FOLDER -ou_f OUTPUT_FOLDER`
<br>

* To reproduce CALTECH101 (5 classes) database results:
`python main.py -NC 5 -DA 3 -N 1415 -NS 600 -NCLA 5 -l_l1 0.01  -l_l2 0.1   -beta 0.001  -K 2000 -M 5  -gamma1 0.5 -gamma2 1.25 -tau 0.00001 -n_w_ite 2000 -l_r 0.05 -n_e_ite 30 -n_i_ite 2000 -d_s 1 -it_sr 30 -nrows 64 -ncols 64 -in_f CALTECH101_5_FOLDER -ou_f OUTPUT_FOLDER`
<br>

* To reproduce CALTECH101 (5 classes) database results:
`python main.py -NC 5 -DA 4 -N 7341 -NS 800 -NCLA 5 -l_l1 0.001 -l_l2 0.01  -beta 0.001  -K 2000 -M 15  -gamma1 0.5 -gamma2 1.25 -tau 0.00001 -n_w_ite 2000 -l_r 0.05 -n_e_ite 30 -n_i_ite 2000 -d_s 1 -it_sr 30 -nrows 64 -ncols 64 -in_f IMAGENET_5_FOLDER -ou_f OUTPUT_FOLDER`
<br>
