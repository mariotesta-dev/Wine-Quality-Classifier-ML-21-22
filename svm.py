'''
   First we need to try different values
   for the hyperparameters gamma, C by trying
   different combinations of (gamma,C)
'''
import numpy as np
import scipy.optimize
from measures import actual_detection_cost, minDCF_given_t, minimum_detection_cost, t_calibrated
import matplotlib.pyplot as plt
from preprocessing import Z_normalization, Z_normalization_test, Gaussianization

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def svm_primal_from_dual(alpha, DTR, LTR, K):

    N = LTR.shape[0]
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    D = np.vstack((DTR, np.ones(N) * K))
    w_hat_star = np.sum(mcol(alpha) * mcol(Z) * D.T, axis=0)

    return w_hat_star


def linear_svm(DTR, LTR, DTE, LTE, K , C, pi_T, balanced = False, s=False):
    x0 = np.zeros(DTR.shape[1])

    bounds = []
    if balanced == False:
        for i in range(DTR.shape[1]):
            bounds.append((0, C))
    elif balanced == True:
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N
        pi_emp_F = n_F / N
        C_T = C * pi_T / pi_emp_T
        C_F = C * (1-pi_T) / pi_emp_F 
        for i in range(DTR.shape[1]):
            if LTR[i] == 1:
                bounds.append((0,C_T))
            else:
                bounds.append((0,C_F))

    DTR_ext = np.vstack([DTR, np.ones((1,DTR.shape[1]))])
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = np.dot(DTR_ext.T, DTR_ext)
    H = mcol(Z) * mrow(Z) * H

    def JDual(alpha):
        Ha = np.dot(H, mcol(alpha))
        aHa = np.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    x,_,_ = scipy.optimize.fmin_l_bfgs_b(LDual, x0, factr=1.0, approx_grad=False, bounds=bounds)

    #once we have computed the dual solution, we can recover the primal solution
    w_hat_star = svm_primal_from_dual(x, DTR, LTR, K)
    w_star = w_hat_star[0:-1] 
    b_star = w_hat_star[-1] 

    S = np.dot(mcol(w_star).T, DTE) + b_star
    S = S.flatten()
    if s == False:
        minDCF = minimum_detection_cost(S, LTE, 0.5, 1, 1)
        return minDCF
    elif s == True:
        #return scores only!
        return S


def single_fold_linearsvm(DTR, LTR, DTE, LTE, K , pi_T):
    Cs = [1e-3, 1e-2, 0.1, 1]

    unbal = []
    bal = []

    plt.figure()
    plt.title("Linear SVM - Single Fold")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale("log")

    print("**** Showing: Linear SVM - Unbalanced/Balanced - Single Fold")
    for c in Cs:
        minDCF_u = linear_svm(DTR, LTR, DTE, LTE, 1, c, 0.5, balanced = False, s = False)
        minDCF_b = linear_svm(DTR, LTR, DTE, LTE, 1, c, 0.5, balanced = True, s = False)
        unbal.append(minDCF_u)
        bal.append(minDCF_b)
    
    plt.plot(Cs,unbal)
    plt.plot(Cs,bal)
    plt.show()  

def kfold_linsvm_wrapper(DTR, LTR, norm = False, gauss = False):
    unbal = []
    bal = []
    Cs = [1e-3, 1e-2, 0.1, 1]

    for c in Cs:
        unbal.append(k_fold_lin(DTR, LTR, 5, c, norm, gauss, balanced=False))
        bal.append(k_fold_lin(DTR, LTR, 5, c, norm, gauss, balanced=True))

    plt.figure()
    plt.title("Linear SVM - K Fold")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale("log")
    plt.plot(Cs,unbal)
    plt.plot(Cs,bal)
    plt.show()


def k_fold_lin(D, L, K, C = 1.0, norm = True, gauss = True, balanced = False ):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(0)

    all_llr = []
    all_labels = []

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

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
            DTR,mu,std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)
        if gauss == True:
            DTR = Gaussianization(DTR, DTR)
            DTE = Gaussianization(DTR, DTE)

        llr = linear_svm(DTR, LTR, DTE, LTE, 1, C, 0.5, balanced, s = True)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    minDCF = minimum_detection_cost(np.concatenate(all_llr), np.concatenate(all_labels), 0.5, 1, 1)
    
    return minDCF

# Compute the kernel dot-product
def kernel(x1, x2, type, d = 0, c = 0, gamma = 0, csi = 1): #csi = 1 --> eps = ksi^2...... c = [0,1]... gamma = [1.0, 10.0]
    
    if type == "poly":
        # Polynomial kernel of degree d
        return (np.dot(x1.T, x2) + c) ** d + csi**2
    elif type == "RBF":
        # Radial Basic Function kernel
        dist = mcol((x1**2).sum(0)) + mrow((x2**2).sum(0)) - 2 * np.dot(x1.T, x2)
        k = np.exp(-gamma * dist) + csi**2
        return k
    

def quad_kernel_svm(DTR, LTR, DTE, LTE, C, c=0,gamma=0,csi=0, s=False, type="poly"):

    x0 = np.zeros(DTR.shape[1])
    d = 2

    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
    pi_emp_T = n_T / N
    pi_emp_F = n_F / N
    
    C_T = C * 0.5 / pi_emp_T
    C_F = C * (1-0.5) / pi_emp_F 

    bounds = [(0,1)] * LTR.size

    for i in range (LTR.size):
        if (LTR[i]==1):
            bounds[i] = (0,C_T)
        else :
            bounds[i] = (0,C_F)
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = None

    if type == "poly":
        H = mcol(Z) * mrow(Z) * ((np.dot(DTR.T, DTR) + c) ** d + csi**2)  #type == poly
    elif type == "RBF":
        dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T, DTR)
        H = np.exp(-gamma * dist) + csi**2
        H = mcol(Z) * mrow(Z) * H

    def JDual(alpha):
        Ha = np.dot(H, alpha.T)
        aHa = np.dot(alpha, Ha)
        a1 = alpha.sum()
        return 0.5 * aHa - a1, Ha - np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return loss, grad
    
    x,_,_ = scipy.optimize.fmin_l_bfgs_b(LDual, x0, factr=0.0, approx_grad=False, bounds=bounds, maxfun=100000, maxiter=100000)

    #we are not able to compute the priaml solution, but we can still
    #compute the scores like that
    S = np.sum((x*Z).reshape([DTR.shape[1],1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis=0)

    '''
    scores = np.zeros(DTE.shape[1])
    for t in range (DTE.shape[1]): #for every sample in evaluation data (xt)
        score=0
        for i in range (DTR.shape[1]): #for every sample in test data
                score+= x[i]*Z[i] * kernel_el(DTR[:,i],DTE[:,t], gamma, csi, c, type)
        scores[t]=score
    '''
    if s == False:
        minDCF = minimum_detection_cost(S.reshape(S.size,), LTE, 0.5, 1, 1)
        return minDCF
    elif s == True:
        #return scores only! (for k-fold approach)
        return S.reshape(S.size,)


def single_fold_quadsvm(DTR, LTR, DTE, LTE):
    Cs = [1e-2, 0.1, 1, 10]
    vals = []
    vals_n = []

    DTR_n, mu, std = Z_normalization(DTR)
    DTE_n = Z_normalization_test(DTE, mu, std)
    
    plt.figure()
    plt.title("Polynomial Kernel SVM - Single Fold")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale("log")

    for C in Cs:
        vals.append(quad_kernel_svm(DTR, LTR, DTE, LTE, C, c = 0, gamma = 0, csi = 1, s = False, type = "poly"))
        vals_n.append(quad_kernel_svm(DTR_n, LTR, DTE_n, LTE, C, c = 0, gamma = 0, csi = 1, s = False, type = "poly"))
    
    plt.plot(Cs,vals)
    plt.plot(Cs,vals_n)
    plt.show()
    
    vals = []
    vals_n = []
    plt.figure()
    plt.title("RBF Kernel - Single Fold")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale("log")

    for C in Cs:
        vals.append(quad_kernel_svm(DTR, LTR, DTE, LTE, C, c = 0, gamma = 1, csi = 1, s = False, type = "RBF"))
        vals_n.append(quad_kernel_svm(DTR_n, LTR, DTE_n, LTE, C, c = 0, gamma = 1, csi = 1, s = False, type = "RBF"))
    
    plt.plot(Cs,vals)
    plt.plot(Cs,vals_n)
    plt.show()


def kfold_quadsvm_wrapper(DTR, LTR, c = 0, gamma = 1, csi = 1, type = "poly"):
    vals = []
    vals_n = []
    Cs = [1e-2, 0.1, 1, 10]

    for C in Cs:
        vals.append(k_fold_quad(DTR, LTR, 5, C, c, gamma, csi, type, norm = False))
        vals_n.append(k_fold_quad(DTR, LTR, 5, C, c, gamma, csi, type, norm = True))
        #vals.append(kfold_quad_svm(DTR, LTR, C, c, gamma, csi, type=type, norm=False))
        #vals_n.append(kfold_quad_svm(DTR, LTR, C, c, gamma, csi, type=type, norm=True))

    plt.figure()
    plt.title("%s SVM - K Fold" % type)
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale("log")
    plt.plot(Cs,vals)
    plt.plot(Cs,vals_n)
    plt.show()

def k_fold_quad(D, L, K, C = 1.0, c = 0, gamma = 2, csi = 1, type = "poly", norm = True ):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(0)

    all_llr = []
    all_labels = []

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

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
            DTR,mu,std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)

        llr = quad_kernel_svm(DTR, LTR, DTE, LTE, C, c, gamma, csi, True, type)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    minDCF = minimum_detection_cost(np.concatenate(all_llr), np.concatenate(all_labels), 0.5, 1, 1)
    
    return minDCF

def k_fold_quad_actDCF(D, L, K, C = 1.0, c = 0, gamma = 2, csi = 1, type = "poly", norm = True ):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(0)

    all_llr = []
    all_labels = []

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

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
            DTR,mu,std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)

        llr = quad_kernel_svm(DTR, LTR, DTE, LTE, C, c, gamma, csi, True, type)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    actDCF = actual_detection_cost(np.hstack(all_llr), np.hstack(all_labels), 0.5, 1, 1)
    
    return actDCF

def k_fold_quad_calibrated(D, L, K, C = 1.0, c = 0, gamma = 2, csi = 1, type = "poly", norm = True ):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(0)

    all_llr = []
    all_labels = []

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

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
            DTR,mu,std = Z_normalization(DTR)
            DTE = Z_normalization_test(DTE, mu, std)

        llr = quad_kernel_svm(DTR, LTR, DTE, LTE, C, c, gamma, csi, True, type)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)
    
    all_llr = np.concatenate(all_llr)
    all_labels = np.concatenate(all_labels)

    np.random.seed(0)
    np.random.shuffle(all_llr)

    np.random.seed(0)
    np.random.shuffle(all_labels)

    N = int(len(all_llr)/2)

    llr1 = all_llr[:N]
    llr2 = all_llr[N:]
    l1 = all_labels[:N]
    l2 = all_labels[N:]

    est_t = t_calibrated(llr1, l1, 0.5, 1, 1)
    minDCF_star = minDCF_given_t(llr2, l2, 0.5, 1, 1, est_t)
    actDCF_star = actual_detection_cost(llr2, l2, 0.5, 1, 1)
    
    return minDCF_star, actDCF_star

def actDCF_eval_poly(DTR,LTR,DTE,LTE,C,c,csi,norm):

    if norm == True:
        #Normalize DTR_raw and grab parameters
        DTR, mu, std_dev = Z_normalization(DTR)
        #Normalize DTE_raw using parameters from train
        DTE = Z_normalization_test(DTE, mu, std_dev)

    s = quad_kernel_svm(DTR, LTR, DTE, LTE, C, c, 0, csi, True, "poly")
    actDCF = actual_detection_cost(s, LTE, 0.5, 1, 1)

    return actDCF

def actDCF_eval_RBF(DTR,LTR,DTE,LTE,C,gamma,csi,norm):

    if norm == True:
        #Normalize DTR_raw and grab parameters
        DTR, mu, std_dev = Z_normalization(DTR)
        #Normalize DTE_raw using parameters from train
        DTE = Z_normalization_test(DTE, mu, std_dev)

    s = quad_kernel_svm(DTR, LTR, DTE, LTE, C, 0, gamma, csi, True, "RBF")
    actDCF = actual_detection_cost(s, LTE, 0.5, 1, 1)

    return actDCF
