import numpy
from measures import actual_detection_cost, minimum_detection_cost
from preprocessing import Gaussianization, Z_normalization, Z_normalization_test, PCA, PCA_test

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def compute_empirical_mean(X):
    return vcol(X.mean(1))

#the Naive Bayes version of the MVG is simply a Gaussian classifer
#where the covariance matrices are diagonal, since the number of features is small
#we can just multiply C with the identity matrix
def compute_diag_empirical_cov(D):
    mu = compute_empirical_mean(D)
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C * numpy.eye(C.shape[0], C.shape[1])

def compute_empirical_cov(D):
    mu = compute_empirical_mean(D)
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

#Computing covariance matrix
def compute_covariance(D):
    mu = D.mean(1)                    #find mean
    DC = D - vcol(mu)                 #centering data
    C = numpy.dot(DC, DC.T) / float(D.shape[1]) #covariance
    
    return C

#In the Tied versin, the ML solution for the covariance matrix is given by the
# empirical within-class covariance matrix
def compute_within_covariance_tied_naive(D,L):
    SW = []
    for i in range(2):
        class_samples = D[:, L==i]
        C = compute_covariance(class_samples)
        SW.append(class_samples.shape[1]*C)
    cov = sum(SW) / float(D.shape[1])
    return cov * numpy.eye(cov.shape[0], cov.shape[1])

#In the Tied versin, the ML solution for the covariance matrix is given by the
# empirical within-class covariance matrix
def compute_within_covariance(D,L):
    SW = []
    for i in range(2):
        class_samples = D[:, L==i]
        C = compute_covariance(class_samples)
        SW.append(class_samples.shape[1]*C)
    return sum(SW) / float(D.shape[1])

#computes estimates of mean and cov for each class, this is done with a generic approach
#instead of computing v0,v1,v2 using [:, LTR == 0...1...2] as done in the previous labs
def compute_classifier_params(DTR, LTR, alg):

    samples = []
    for label in list(set(LTR)):
        v = DTR[:, LTR == label]
        samples.append(v)
    
    params = []
    for v in samples:
        mu_ML = compute_empirical_mean(v)
        if(alg == 'naive'):
            C_ML = compute_diag_empirical_cov(v)
        elif(alg == 'mult'):
            C_ML = compute_empirical_cov(v)
        elif(alg == 'tied-cov'):
            C_ML = compute_within_covariance(DTR,LTR)
        elif(alg == 'tied-naive'):
            C_ML = compute_within_covariance_tied_naive(DTR,LTR)
        params.append([mu_ML,C_ML])
    
    return params

#logpdf_GAU_ND_Opt = takes X ([M x N] dataset) as an argument,
#then computes the vector of log densities for feature vector x (so, for each column of X).
def logpdf_GAU_ND_Opt(X,mu,C):
    C_inv = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]
    #the logarithmic determinant of C (covariance of X) can be computed using the slodget function:
    #it returns [0]= sign of determinand, [1]=abs value of logarithmic determinant
    
    Y = [] #vector of log-densities for each column of X: Y = [log N (x_1|µ, Σ),. . . ,log N (x_n |µ, Σ)]
    for i in range(X.shape[1]):
        x = X[:, i:i+1] #takes one column at a time
        res = const + -0.5 * numpy.dot( (x-mu).T, numpy.dot(C_inv, (x-mu)) )
        Y.append(res)
    return numpy.array(Y).ravel() #flatten and returns a 1-D array

def class_loglikelihood(DTE, classifier_params):
    S = []
    for params in classifier_params:
        class_loglike = logpdf_GAU_ND_Opt(DTE, params[0], params[1])
        S.append(class_loglike)
    return numpy.array(S)

#computes vector of all densities of dataset X
def pdfND(X,mu,C):
    return numpy.exp(logpdf_GAU_ND_Opt(X,mu,C))

def correct_predictions(predicted_labels, LTE):
    correct = 0
    for i,x in enumerate(predicted_labels):
        if x == LTE[i]:
            correct += 1
    return correct

def gaussian_classifier_loglikeratio(DTR,LTR,DTE,alg="mult"):
    #alg can be: 'mult', 'naive', 'tied-cov', 'tied-naive'
    classifier_params = compute_classifier_params(DTR,LTR,alg)
    S = class_loglikelihood(DTE,classifier_params)
    llr = S[1] - S[0]
    return llr #return the loglikelihood ratio

def single_fold_gaussian(DTR,LTR,DTE,LTE,pi, Cfn, Cfp, alg="mult", k=None):

    llr = gaussian_classifier_loglikeratio(DTR, LTR, DTE, alg)
    _minDCF = minimum_detection_cost(llr, LTE, pi, Cfn, Cfp)

    return llr, _minDCF

#alg: ['mult', 'naive', 'tied-cov', 'tied-naive']
def k_fold_gaussian(D, L, K, pi, Cfp, Cfn, alg="mult", gauss=False, pca=False):

    sizePartitions = int(D.shape[1]/K)
    numpy.random.seed(0)

    all_llr = []
    all_labels = []

    # permutate the indexes of the samples
    idx_permutation = numpy.random.permutation(D.shape[1])

    # put the indexes inside different partitions
    idx_partitions = []
    for i in range(0, D.shape[1], sizePartitions):
        idx_partitions.append(list(idx_permutation[i:i+sizePartitions]))

    # for each fold, consider the ith partition in the test set
    # the other partitions in the train set
    for i in range(K):
        # keep the i-th partition for test
        # keep the other partitions for train
        idx_test = idx_partitions[i]
        idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

        # from lists of lists collapse the elemnts in a single list
        idx_train = sum(idx_train, [])

        # partition the data and labels using the already partitioned indexes
        DTR = D[:, idx_train]
        DTE = D[:, idx_test]
        LTR = L[idx_train]
        LTE = L[idx_test]

        if gauss:
            DTR,mu,std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)
            DTR = Gaussianization(DTR,DTR)
            DTE = Gaussianization(DTR, DTE)
        if pca is not False:
            DTR,proj = PCA(DTR,pca)
            DTE = PCA_test(DTE, proj)

        llr = gaussian_classifier_loglikeratio(DTR, LTR, DTE, alg)
        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    minDCF = minimum_detection_cost(numpy.concatenate(all_llr), numpy.concatenate(all_labels), pi, Cfn, Cfp)
    
    return minDCF

