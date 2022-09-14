from utils import vcol
import numpy
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
    Z-normalization (or standardization) subtract to each feature X its mean and divides by the standard deviation.
    In order to have a standard distribution with mean = 0 and variance = 1
'''
def Z_normalization(D):
    mu = vcol(D.mean(1))
    std_dev = vcol(numpy.std(D, axis=1))
    N = (D-mu)/std_dev
    return N, mu, std_dev

def Z_normalization_test(D,mu,std_dev):
    N = (D-mu)/std_dev
    return N

'''
    Gaussianization consists in mapping the features to values with uniform distribution and then transforming them
    through the inverse of the Gaussian cumulative distribution function. 
    First, we compute the rank of a feature x over the training set
    then the transformed feature is computed using the Inverse of the Cumulative Distribution Function
'''

'''
    With gaussianization, first we compute the rank of a feature x over the training set,
    then the transformed feature is computed using the Inverse of the Cumulative Distribution Function (or percent point function norm.ppf)
    This will return a value (that functions as a 'standard-deviation multiplier') marking where 95% of data points would be contained if our data is a normal distribution.
'''
def Gaussianization(TD, D):
    N = TD.shape[1]
    ranks = []
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum += (D[j, :] < TD[j, i]).astype(int)
        tempSum += 1
        ranks.append(tempSum / (N + 2))
    y = norm.ppf(ranks)
    return y

#Computing covariance matrix
def compute_covariance(D):
    mu = D.mean(1)                    #find mean, mu is a 1-D array and needs to be reshaped into a col vector
    DC = D - vcol(mu)                 #centering data (removing mean from all the points)
    C = numpy.dot(DC, DC.T) / float(D.shape[1]) #covariance
    
    return C

def PCA(D,m):
    C = compute_covariance(D)
    #compute eigenvectors and eigenvalues using numpy function
    #which returns the eigenvalues (s), sorted from smallest to largest,
    # and the corresponding eigenvectors (columns of U)
    #s, U1 = numpy.linalg.eigh(C)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    DP = numpy.dot(P.T, D) #apply projection
    
    return DP,P

def PCA_test(D, P):
    DP = numpy.dot(P.T, D)
    return DP

def scree_plot(D):

    C = compute_covariance(D)

    U, s, Vh = numpy.linalg.svd(C)
    eigen_values = s
    sing_vals=numpy.arange(len(eigen_values)) + 1
    plt.plot(sing_vals,eigen_values, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue') 
    plt.show() 
