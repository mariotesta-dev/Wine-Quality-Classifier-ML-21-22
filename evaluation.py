from preprocessing import Gaussianization, Z_normalization, Z_normalization_test, PCA, PCA_test
from gaussian_models import single_fold_gaussian
from logistic_regression import single_fold_logistic_regression, single_fold_quadratic_logistic_regression, actDCF_eval_quadlogreg
from svm import actDCF_eval_RBF, actDCF_eval_poly, linear_svm, quad_kernel_svm
from gmm import eval_wrapper

MVG = False
LOG = False
SVM = False
GMM = True

def evaluation(DTR_raw, LTR, DTE_raw, LTE):

    apps = [
        {'pi':0.5, 'Cfn':1, 'Cfp':1},
        ]

    # ****** Classification ******
    algs = ['mult', 'naive', 'tied-cov', 'tied-naive']

    #Normalize DTR_raw and grab parameters
    DTR_n, mu, std_dev = Z_normalization(DTR_raw)
    #Normalize DTE_raw using parameters from train
    DTE_n = Z_normalization_test(DTE_raw, mu, std_dev)

    #Gaussianize DTR_n
    DTR_g = Gaussianization(DTR_n,DTR_n)
    #Gaussianize DTE_n using DTR_n
    DTE_g = Gaussianization(DTR_n, DTE_n)

    #Apply PCA(10) to DTR_g and grab projection
    DTR_g_p10, proj10 = PCA(DTR_g,10)
    #Apply PCA(10) to DTE_g using DTR_g projection
    DTE_g_p10 = PCA_test(DTE_g, proj10)

    #Apply PCA(9) to DTR_g and grab projection
    DTR_g_p9, proj9 = PCA(DTR_g,9)
    #Apply PCA(9) to DTE_g using DTR_g projection
    DTE_g_p9 = PCA_test(DTE_g, proj9)
    

    for app in apps:
        print(app)

        # ****** Gaussian Models ******
        if MVG == True:
            for alg in algs:
                print("**** %s ****" % alg)
                # ****** Raw - NO PCA ******
                llr, _minDCF = single_fold_gaussian(DTR_raw, LTR, DTE_raw, LTE, app['pi'], app['Cfn'], app['Cfp'], alg, k=None)
                print(_minDCF)
                # ****** Gauss - NO PCA ******
                llr, _minDCF = single_fold_gaussian(DTR_g, LTR, DTE_g, LTE, app['pi'], app['Cfn'], app['Cfp'], alg, k=None)
                print(_minDCF)
                # ****** Gauss - PCA = 10 ******
                llr, _minDCF = single_fold_gaussian(DTR_g_p10, LTR, DTE_g_p10, LTE, app['pi'], app['Cfn'], app['Cfp'], alg, k=None)
                print(_minDCF)
                # ****** Gauss - PCA = 9 ******
                llr, _minDCF = single_fold_gaussian(DTR_g_p9, LTR, DTE_g_p9, LTE, app['pi'], app['Cfn'], app['Cfp'], alg, k=None)
                print(_minDCF)
                print("\n")

    
    # ****** Logistic Regression ******
    if LOG == True:
        
        print("**** Logistic Regression ****")
            
        print("without balancing RAW")
        #without balancing RAW
        for l in [1e-3]:
            score, _minDCF = single_fold_logistic_regression(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-3]:
            score, _minDCF = single_fold_logistic_regression(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("**** Quadratic Logistic Regression ****")
        
        # ****** Quad Logistic Regression ******
        print("without balancing RAW")
        #without balancing RAW
        for l in [1e-3]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-3]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
    
    if SVM == True:
        print(linear_svm(DTR_n, LTR, DTE_n, LTE, 1, 0.1 , 0.5, balanced = False, s = False))
        print(quad_kernel_svm(DTR_n, LTR, DTE_n, LTE, 0.1, c=1, gamma=0, csi=1, s=False, type="poly"))
        print(quad_kernel_svm(DTR_n, LTR, DTE_n, LTE, C = 1.0, c = 1, gamma = 1, csi = 1, s = False, type = "RBF"))

    if GMM == True:
        eval_wrapper(DTR_raw, LTR, DTE_raw, LTE, covariance_type="Full")
        eval_wrapper(DTR_raw, LTR, DTE_raw, LTE, covariance_type="Diagonal")
        eval_wrapper(DTR_raw, LTR, DTE_raw, LTE, covariance_type="Tied")
        eval_wrapper(DTR_raw, LTR, DTE_raw, LTE, covariance_type="Tied Diagonal")

def actDCFeval(D,L,TEST,L_TEST):
    #print(actDCF_eval_quadlogreg(D, L, TEST, L_TEST, 1e-3, pi = 0.5, Cfn = 1, Cfp = 1, norm = True))
    #print(actDCF_eval_poly(D, L, TEST, L_TEST, C = 0.1, c = 1, csi = 1, norm = True))
    print(actDCF_eval_RBF(D,L,TEST,L_TEST, C = 1.0, gamma = 1, csi = 1, norm = True))