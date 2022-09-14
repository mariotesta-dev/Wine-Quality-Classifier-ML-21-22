

import numpy
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt
from measures import minimum_detection_cost
import preprocessing
from utils import load_data

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))


def logpdf_GAU_ND(x, mu, C):
    #x = samples
    #mu= mean
    #C = covariance matrix
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)
    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()

def logpdf_GMM(X, GMM):
    #x = samples, matrix of shape (D=size of a sample, N= number of samples)
    # gmm = [(w1,mu1, C1), (w2,mu2,C2),...]
    S= compute_joint_log_densities(X,GMM)
    
    #compute log marginal log density
    marginal_logdens = scipy.special.logsumexp(S, axis=0) #shape(N,), the i-th component contains the log density for sample xi
    return marginal_logdens

def compute_joint_log_densities(X, gmm):
    N_samples = X.shape[1]
    M_components= len(gmm)
    S = numpy.zeros((M_components, N_samples))

    for j in range (M_components):
        mean=gmm[j][1]
        covariance=gmm[j][2]
        S[j,:] = logpdf_GAU_ND(X, mean, covariance)
        weight=gmm[j][0]
        S[j, :] += numpy.log(weight)
    return S


def compute_EM_algorithm(X, initial_GMM, threshold = 10**(-6), constrained=False , psi=-1, covariance_type="full"):
    # gmm = [(w1,mu1, C1), (w2,mu2,C2),...]
    estimated_gmm = initial_GMM.copy()
    first = True


    while(True):
        # E STEP
        joint_log_densities = compute_joint_log_densities(X, estimated_gmm) #matrix S (num_gmm,num_data)
        marginal_log_densities = logpdf_GMM(X, estimated_gmm) # (num_data,)
        log_posterior_distribution= joint_log_densities - marginal_log_densities
        posterior_distributions = numpy.exp(log_posterior_distribution) # responsabilities (1 for each component and for each sample = num_gmm x num_samples)
        
        avg_log_likelihood_nuova = numpy.average(marginal_log_densities)
        
        if (first):
            first=False
        else:
            if (avg_log_likelihood_nuova < avg_log_likelihood_vecchia):
                print("Attenzione, errore")
            if (avg_log_likelihood_nuova - avg_log_likelihood_vecchia < threshold):
                #print("average log likelihood:", avg_log_likelihood_vecchia)
                break
        
        avg_log_likelihood_vecchia = avg_log_likelihood_nuova
        
        #M step -> update the model parameter

        Zg = posterior_distributions.sum(axis=1) #1 number for each of the GMM components
        Fg = []
        for i in range (posterior_distributions.shape[0]): #1 for each component
            Fg.append((posterior_distributions[i:i+1, :]*X).sum(axis=1))
        
        Sg= []
        for i in range (posterior_distributions.shape[0]):
            Sg.append(0) 
            for j in range (X.shape[1]):
                Sg[i]+=posterior_distributions[i][j]*numpy.dot(X[:, j:j+1],X[:, j:j+1].T )
        
        tied_cov=0
        for i in range (len(estimated_gmm)):
            newM = mcol(Fg[i] / Zg[i])
            newS = Sg[i] / Zg[i] - numpy.dot(newM, newM.T) #4x4 (4= dimensione sample)
            
            if (covariance_type=="Diagonal" or covariance_type=="Tied Diagonal"): 
                newS = newS*numpy.eye(newS.shape[0]) #only the diagonal
            
            if (covariance_type=="Tied" or covariance_type=="Tied Diagonal") : 
                tied_cov += newS*Zg[i]
            
            if ((covariance_type!="Tied" or covariance_type=="Tied Diagonal") and constrained): newS = newCov_constrained(newS, psi)
            
            newW = Zg[i] / Zg.sum()
            estimated_gmm[i] = (newW, newM, newS)
        
        if (covariance_type=="Tied" or covariance_type=="Tied Diagonal"):
            tied_cov=tied_cov/X.shape[1]
            if (constrained): tied_cov=newCov_constrained(tied_cov, psi)
            for i in range (len(estimated_gmm)):
                estimated_gmm[i]= (estimated_gmm[i][0], estimated_gmm[i][1], tied_cov)


    return (estimated_gmm)


def compute_LBG_algorithm(X, number_components, constrained=False , psi=-1, covariance_type="Full", alpha=0.1, delta_l=10**(-6)):

    x = X
    mean = x.mean(1).reshape((x.shape[0], 1))

    X_centered = x - mean
    cov =  numpy.dot(X_centered, X_centered.T)/ x.shape[1]
    
    if (covariance_type =="Diagonal" or covariance_type =="Tied Diagonal"): cov=cov*numpy.eye(cov.shape[0])
    if (constrained): cov=newCov_constrained(cov, psi)
    
    initial_gmm = [(1.0, mean, cov )]
    
    estimated_GMM = initial_gmm
    while (len(estimated_GMM)<number_components):
        estimated_GMM = split2GMM(estimated_GMM, alpha)
        estimated_GMM = compute_EM_algorithm(x, initial_GMM= estimated_GMM, threshold=delta_l, constrained=constrained,psi= psi, covariance_type=covariance_type)
    return estimated_GMM

def split2GMM(gmm, alpha= 0.1):
    splittedGmm = []
    for component in gmm:
        w = component[0]
        mu = component[1]
        sigma = component[2]
        U, s, Vh = numpy.linalg.svd(sigma)
        d = U[:,0:1] * s[0]**0.5 * alpha
        newComp1= (w/2, mu+d, sigma)
        newComp2= (w/2, mu-d, sigma)

        splittedGmm.append(newComp1)
        splittedGmm.append(newComp2)
        
    return splittedGmm

def newCov_constrained (cov, psi):
    U,s,_ =numpy.linalg.svd(cov)
    s[s<psi]=psi
    covNew= numpy.dot(U, mcol(s) * U.T)
    return covNew


def GMM_computeLogLikelihoodRatios(DTR, LTR, DTE, otherParams): #otherparams= [constrained, psi, covariance_type, alpha, number_components, delta_l]
    constrained= otherParams[0]
    psi=otherParams[1]
    covariance_type=otherParams[2]
    alpha=otherParams[3]
    number_components=otherParams[4]
    delta_l=otherParams[5]

    #train a gmm for each class
    n_classes = len(numpy.unique(LTR))
    gmm_classes = []
    log_class_conditional_distribution= numpy.zeros((n_classes, DTE.shape[1]))
    #obtain one gmm for each class
    for i in range (n_classes):
        data = DTR[:, LTR == i]
        gmm_classes.append(compute_LBG_algorithm(data, number_components= number_components, constrained=constrained, psi=psi, covariance_type= covariance_type,alpha= alpha, delta_l=delta_l))
        log_class_conditional_distribution[i,:]=mrow(logpdf_GMM(GMM=gmm_classes[i], X= DTE))
     
    llr = log_class_conditional_distribution[1,:] - log_class_conditional_distribution[0,:]
    return llr #loglikelihoodRatio
 

# Perform k-fold cross validation on test data for the specified model
def k_fold_cross_validation(D, L, k, otherParams, gauss = False):

    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    start_index = 0
    elements = int(D.shape[1] / k)

    llr = numpy.zeros([D.shape[1], ])

    for count in range(k):
        # Define training and test partitions
        if start_index + elements > D.shape[1]:
            end_index = D.shape[1]
        else:
            end_index = start_index + elements 

        idxTrain = numpy.concatenate((idx[0:start_index], idx[end_index:]))
        idxTest = idx[start_index:end_index]

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
    
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        if gauss:
            DTR,m,s = preprocessing.Z_normalization(DTR)
            DTE = preprocessing.Z_normalization_test(DTE, m, s)
            DTR = preprocessing.Gaussianization(DTR,DTR)
            DTE = preprocessing.Gaussianization(DTR,DTE)

        # Train the classifier and compute llr on the current partition
        llr[idxTest] = GMM_computeLogLikelihoodRatios(DTR, LTR, DTE, otherParams)
        start_index += elements

    return llr

def wrapper(DTR, LTR, covariance_type):
    gmm_comp = [1,2,4,8,16]

    vals = []
    vals_g = []

    constrained=True
    psi=0.01
    alpha=0.1
    delta_l=10**(-6)

    print(covariance_type)

    for i in range(len(gmm_comp)):
        params = [constrained, psi, covariance_type, alpha, gmm_comp[i], delta_l]
        print("-------> working on raw data, comp= ", gmm_comp[i])
        llr = k_fold_cross_validation(DTR, LTR, 5, params, gauss=False)
        mindcf = minimum_detection_cost(llr, LTR, 0.5, 1, 1)
        print(mindcf)
        vals.append(mindcf)

        print("-------> working on gaussianized data, comp= ", gmm_comp[i])
        llr = k_fold_cross_validation(DTR, LTR, 5, params, gauss=True)
        mindcf = minimum_detection_cost(llr, LTR, 0.5, 1, 1)
        print(mindcf)
        vals_g.append(mindcf)


def plot_GMM_results():
    
    gmm_comp = [1,2,4,8,16]

    full_raw = [0.317, 0.312, 0.303, 0.289, 0.302]
    diag_raw = [0.393, 0.404, 0.350, 0.329, 0.316]
    tied_raw = [0.317, 0.317, 0.312, 0.314, 0.304]
    tied_diag_raw = [0.393, 0.415, 0.361, 0.363, 0.324]

    full_g = [0.306, 0.343, 0.407, 0.411, 0.429]
    diag_g = [0.425, 0.393, 0.313, 0.331, 0.360]
    tied_g = [0.306, 0.306, 0.321, 0.327, 0.359]
    tied_diag_g = [0.424, 0.417, 0.362, 0.318, 0.330]

    '''
    for t in ["Raw", "Gaussianized"]:
        for alg in ["Full Covariance", "Diagonal Covariance", "Tied-Full Covariance", "Tied-Diag Covariance"]:
    '''            

    widthbar = 0.2

    x_ind = numpy.arange(len(gmm_comp))

    raw_ind = x_ind - widthbar/2
    gau_ind = x_ind + widthbar/2

    lb1 = "minDCF (prior=0.5) - Raw"
    lb2 = "minDCF (prior=0.5) - Gaussianized"
        
    plt.figure()
    plt.bar(raw_ind, full_raw, width = widthbar, color = 'orange', label = lb1)
    plt.bar(gau_ind, full_g, width = widthbar, color = 'red', label = lb2)
    plt.title("Full Covariance")
    plt.xticks(x_ind ,gmm_comp)
    plt.ylabel('minDCFs')
    plt.xlabel('GMM components')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.bar(raw_ind, diag_raw, width = widthbar, color = 'orange', label = lb1)
    plt.bar(gau_ind, diag_g, width = widthbar, color = 'red', label = lb2)
    plt.title("Diagonal Covariance")
    plt.xticks(x_ind ,gmm_comp)
    plt.ylabel('minDCFs')
    plt.xlabel('GMM components')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.bar(raw_ind, tied_raw, width = widthbar, color = 'orange', label = lb1)
    plt.bar(gau_ind, tied_g, width = widthbar, color = 'red', label = lb2)
    plt.title("Tied Covariance")
    plt.xticks(x_ind ,gmm_comp)
    plt.ylabel('minDCFs')
    plt.xlabel('GMM components')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    plt.bar(raw_ind, tied_diag_raw, width = widthbar, color = 'orange', label = lb1)
    plt.bar(gau_ind, tied_diag_g, width = widthbar, color = 'red', label = lb2)
    plt.title("Tied-Diag Covariance")
    plt.xticks(x_ind ,gmm_comp)
    plt.ylabel('minDCFs')
    plt.xlabel('GMM components')
    plt.legend(loc="lower left")
    plt.show()

def eval_wrapper(DTR, LTR, DTE, LTE, covariance_type="Full"):
    gmm_comp = [1,2,4,8,16]

    DTR_n, m, s = preprocessing.Z_normalization(DTR)
    DTE_n = preprocessing.Z_normalization_test(DTE, m, s)
    DTR_g = preprocessing.Gaussianization(DTR_n,DTR_n)
    DTE_g = preprocessing.Gaussianization(DTR_n, DTE_n)

    vals = []
    vals_g = []

    constrained=True
    psi=0.01
    alpha=0.1
    delta_l=10**(-6)

    print(covariance_type)

    for i in range(len(gmm_comp)):
        params = [constrained, psi, covariance_type, alpha, gmm_comp[i], delta_l]
        print("-------> working on raw data, comp= ", gmm_comp[i])
        llr = GMM_computeLogLikelihoodRatios(DTR, LTR, DTE, params)
        mindcf = minimum_detection_cost(llr, LTE, 0.5, 1, 1)
        print(mindcf)
        vals.append(mindcf)

        print("-------> working on gaussianized data, comp= ", gmm_comp[i])
        llr = GMM_computeLogLikelihoodRatios(DTR_g, LTR, DTE_g, params)
        mindcf = minimum_detection_cost(llr, LTE, 0.5, 1, 1)
        print(mindcf)
        vals_g.append(mindcf)


if __name__ == '__main__':
    
    DTR,LTR = load_data("data/Train.txt")

    wrapper(DTR, LTR, covariance_type="Full")
    wrapper(DTR, LTR, covariance_type="Diagonal")
    wrapper(DTR, LTR, covariance_type="Tied")
    wrapper(DTR, LTR, covariance_type="Tied Diagonal")

    #due to computational time reasons, results have been saved and manually inserted into arrays to be plotted
    plot_GMM_results()