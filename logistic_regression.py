import numpy
import scipy.optimize
from measures import actual_detection_cost, minimum_detection_cost
from preprocessing import Z_normalization, Z_normalization_test

def vcol(v): #v is a row vector(1, v.size), the function returns a column vector (v.size,1)
    return v.reshape((v.size, 1))

def vrow(v): #v is a column vector(v.size,1), the function returns a row vector (1,v.size)
    return v.reshape((1,v.size))

def correct_predictions(predicted_labels, LTE):
    correct = 0
    for i,x in enumerate(predicted_labels):
        if x == LTE[i]:
            correct += 1
    return correct

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        S = numpy.dot(w.T, DTR) + b
        Z = 2 * LTR - 1
        J = l / 2 * numpy.linalg.norm(w)**2 + numpy.logaddexp(0,-S*Z).mean()
        return J

    return logreg_obj

def logreg_obj_wrap_balanced(DTR, LTR, l, pi_T):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]

        Nt = sum(LTR == 1)
        Nf = sum(LTR == 0)

        J = l/2 * numpy.linalg.norm(w)**2 + pi_T / Nt * sum(numpy.log1p(numpy.exp( - (numpy.dot(w.T, DTR[:, LTR == 1]) + b )))) + \
            (1 - pi_T) / Nf * sum(numpy.log1p(numpy.exp((numpy.dot(w.T, DTR[:, LTR == 0]) + b ))))

        return J

    return logreg_obj

def logistic_regression(DTR,LTR,DTE,l):

    logreg_obj = logreg_obj_wrap(DTR, LTR,l)
    x0 = numpy.zeros(DTR.shape[0]+1)
    v,J,d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w = v[0:DTR.shape[0]]
    b = v[-1]
    S = numpy.dot(w.T, DTE) + b
        
    return S

def logistic_regression_balanced(DTR,LTR,DTE,l,pi_T):

    logreg_obj = logreg_obj_wrap_balanced(DTR, LTR,l,pi_T)
    x0 = numpy.zeros(DTR.shape[0]+1)
    v,J,d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w = v[0:DTR.shape[0]]
    b = v[-1]
    S = numpy.dot(w.T, DTE) + b
        
    return S

# Map features to the quadratic feature space
def map_to_feature_space(D):
    phi = numpy.zeros([D.shape[0]**2+D.shape[0], D.shape[1]])   #transformed space of dimension D^2 + D
    for index in range(D.shape[1]):
        x = D[:, index].reshape(D.shape[0], 1)
        # phi = [vec(x*x^T), x]^T
        phi[:, index] = numpy.concatenate((numpy.dot(x, x.T).reshape(x.shape[0]**2, 1), x)).reshape(phi.shape[0],)
    return phi

# Train a quadratic logistic regression model and evaluate it on test data
def quadratic_logistic_regression(DTR, LTR, DTE, l):

    # Map training features to expanded feature space
    phi = map_to_feature_space(DTR)

    # Train a linear regression model on expanded feature space
    logreg_obj = logreg_obj_wrap(phi, LTR, l)
    optV, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(phi.shape[0] + 1), approx_grad = True)   
    w, b = optV[0:phi.shape[0]], optV[-1]

    # Map test features to expanded feature space
    phi_test = map_to_feature_space(DTE)

    # Compute scores
    s = numpy.dot(w.T, phi_test) + b

    return s

def quadratic_logistic_regression_balanced(DTR, LTR, DTE, l, pi_T):

    # Map training features to expanded feature space
    phi = map_to_feature_space(DTR)

    # Train a linear regression model on expanded feature space
    logreg_obj = logreg_obj_wrap_balanced(phi, LTR, l, pi_T)
    optV, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(phi.shape[0] + 1), approx_grad = True)   
    w, b = optV[0:phi.shape[0]], optV[-1]

    # Map test features to expanded feature space
    phi_test = map_to_feature_space(DTE)

    # Compute scores
    s = numpy.dot(w.T, phi_test) + b

    return s

def single_fold_logistic_regression(DTR,LTR,DTE,LTE,l, pi, Cfn, Cfp,):

    llr = logistic_regression(DTR,LTR,DTE,l)
    _minDCF = minimum_detection_cost(llr,LTE,pi,Cfn,Cfp)

    return llr, _minDCF

def single_fold_logistic_regression_balanced(DTR,LTR,DTE,LTE,l, pi, pi_T, Cfn, Cfp):

    S = logistic_regression_balanced(DTR,LTR,DTE,l,pi_T)
    _minDCF = minimum_detection_cost(S,LTE,pi,Cfn,Cfp)

    return S, _minDCF

def single_fold_quadratic_logistic_regression(DTR,LTR,DTE,LTE,l, pi, Cfn, Cfp, balanced=False):

    if balanced==False:
        
        llr = quadratic_logistic_regression(DTR,LTR,DTE,l)
        _minDCF = minimum_detection_cost(llr,LTE,pi,Cfn,Cfp)

        return llr, _minDCF
    elif balanced==True:
       
        llr = quadratic_logistic_regression_balanced(DTR,LTR,DTE,l,0.5)
        _minDCF = minimum_detection_cost(llr,LTE,pi,Cfn,Cfp)

        return llr, _minDCF

def log_reg(DTR, LTR, DTE, l, type="lin-ub"):
    if type=="lin-ub":
        return logistic_regression(DTR, LTR, DTE, l)
    elif type=="lin-b":
        return logistic_regression_balanced(DTR, LTR, DTE, l, 0.5)
    if type=="quad-ub":
        return quadratic_logistic_regression(DTR, LTR, DTE, l)
    elif type=="quad-b":
        return quadratic_logistic_regression_balanced(DTR, LTR, DTE, l, 0.5)


def k_fold_log_reg(D, L, K, l, pi, Cfp, Cfn, type="lin-ub", norm=False):

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

        if norm == True:
            DTR, mu, std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)

        # Train the classifier and compute llr on the current partition
        llr = log_reg(DTR, LTR, DTE, l, type)
        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    minDCF = minimum_detection_cost(numpy.concatenate(all_llr), numpy.concatenate(all_labels), pi, Cfn, Cfp)

    return minDCF

def k_fold_log_reg_actDCF(D, L, K, l, pi, Cfp, Cfn, type="lin-ub", norm=False):

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

        if norm == True:
            DTR, mu, std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)

        # Train the classifier and compute llr on the current partition
        llr = log_reg(DTR, LTR, DTE, l, type)
        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    actDCF = actual_detection_cost(numpy.concatenate(all_llr), numpy.concatenate(all_labels), pi, Cfn, Cfp)

    return actDCF


def actDCF_eval_quadlogreg(DTR, LTR, DTE, LTE, lam, pi, Cfn = 1, Cfp = 1, norm = True):
    if norm == True:
        #Normalize DTR_raw and grab parameters
        DTR, mu, std_dev = Z_normalization(DTR)
        #Normalize DTE_raw using parameters from train
        DTE = Z_normalization_test(DTE, mu, std_dev)

    llr = quadratic_logistic_regression(DTR,LTR,DTE,lam)
    actDCF = actual_detection_cost(llr, LTE, pi, Cfn, Cfp)

    return actDCF
