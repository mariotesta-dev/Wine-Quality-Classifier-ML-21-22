from utils import load_data, plot_heatmap_pearson, dataset_test_stats, plot_histogram, split_db_2to1, kfold
from preprocessing import Gaussianization, Z_normalization, Z_normalization_test, PCA, PCA_test
from gaussian_models import single_fold_gaussian, k_fold_gaussian
from logistic_regression import single_fold_logistic_regression, single_fold_logistic_regression_balanced, single_fold_quadratic_logistic_regression, k_fold_log_reg, k_fold_log_reg_actDCF
from svm import single_fold_linearsvm, kfold_linsvm_wrapper, single_fold_quadsvm, kfold_quadsvm_wrapper, k_fold_quad_actDCF
from gmm import wrapper
from evaluation import evaluation, actDCFeval

VALIDATION = False
EVALUATION = True

MVG = False
LOG = False
SVM = False
GMM = False

def single_fold(DTR, LTR):

    apps = [
        {'pi':0.5, 'Cfn':1, 'Cfp':1},
        {'pi':0.9, 'Cfn':1, 'Cfp':1},
        {'pi':0.1, 'Cfn':1, 'Cfp':1}
        ]

    # ****** Classification ******
    algs = ['mult', 'naive', 'tied-cov', 'tied-naive']

    '''
        SINGLE FOLD: 2/3 Train Set - 1/3 Validation Set 
    '''
    #Raw dataset
    (DTR_raw,LTR),(DTE_raw,LTE) = split_db_2to1(DTR,LTR)

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
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_logistic_regression(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_logistic_regression(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)

        print("with balancing RAW")
        #with balancing
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_logistic_regression_balanced(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], 0.5, apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)

        print("with balancing Normalized")
        #with balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_logistic_regression_balanced(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], 0.5, apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("**** Quadratic Logistic Regression ****")
        
        # ****** Quad Logistic Regression ******
        print("without balancing RAW")
        #without balancing RAW
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'])
            print(l, _minDCF)
        
        print("with balancing RAW")
        #with balancing
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_raw, LTR, DTE_raw, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'], balanced=True)
            print(l, _minDCF)

        print("with balancing Normalized")
        #with balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            score, _minDCF = single_fold_quadratic_logistic_regression(DTR_n, LTR, DTE_n, LTE, l, apps[0]['pi'], apps[0]['Cfn'], apps[0]['Cfp'], balanced=True)
            print(l, _minDCF)
    
    if SVM == True:
        #find C by plotting graphs:
        single_fold_linearsvm(DTR_raw, LTR, DTE_raw, LTE, 1, 0.5) #raw
        single_fold_linearsvm(DTR_n, LTR, DTE_n, LTE, 1, 0.5)     #norm
        single_fold_linearsvm(DTR_g, LTR, DTE_g, LTE, 1, 0.5)     #gauss

        single_fold_quadsvm(DTR_raw, LTR, DTE_raw, LTE)

def kfold(DTR, LTR, k = 5):
    
    #DTR and LTR will be splitted into folds (and the folds will eventually be processed with Norm, Gauss, PCA, etc)
    #inside each algorithm function

    apps = [
        {'pi':0.5, 'Cfn':1, 'Cfp':1},
        {'pi':0.9, 'Cfn':1, 'Cfp':1},
        {'pi':0.1, 'Cfn':1, 'Cfp':1}
        ]

    # ****** Classification ******
    algs = ['mult', 'naive', 'tied-cov', 'tied-naive']

    for app in apps:
        print(app)
        if MVG == True:
                for alg in algs:
                    print("**** %s ****" % alg)
                    # ****** Raw - NO PCA ******
                    _minDCF = k_fold_gaussian(DTR, LTR, 5, app['pi'], app['Cfn'], app['Cfp'], alg, gauss=False, pca=False)
                    print(_minDCF)
                    # ****** Gauss - NO PCA ******
                    _minDCF = k_fold_gaussian(DTR, LTR, 5, app['pi'], app['Cfn'], app['Cfp'], alg, gauss=True, pca=False)
                    print(_minDCF)
                    # ****** Gauss - PCA = 10 ******
                    _minDCF = k_fold_gaussian(DTR, LTR, 5, app['pi'], app['Cfn'], app['Cfp'], alg, gauss=True, pca=10)
                    print(_minDCF)
                    # ****** Gauss - PCA = 9 ******
                    _minDCF = k_fold_gaussian(DTR, LTR, 5, app['pi'], app['Cfn'], app['Cfp'], alg, gauss=True, pca=9)
                    print(_minDCF)
                    print("\n")
    
        # ****** Logistic Regression ******
    if LOG == True:
        
        print("**** Logistic Regression ****")
            
        print("without balancing RAW")
        #without balancing RAW
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "lin-ub", norm=False)
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "lin-ub", norm=True)
            print(l, _minDCF)

        # BEWARE: kfold + balancing took me a lot to compute results, so they are commented in order to reduce waiting time!!!
        '''
        print("with balancing RAW")
        #with balancing
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "lin-b", norm=False)
            print(l, _minDCF)

        print("with balancing Normalized")
        #with balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "lin-b", norm=True)
            print(l, _minDCF)
        '''
        
        print("**** Quadratic Logistic Regression ****")
        
        # ****** Quad Logistic Regression ******
        print("without balancing RAW")
        #without balancing RAW
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "quad-ub", norm=False)
            print(l, _minDCF)
        
        print("without balancing Normalized")
        #without balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "quad-ub", norm=True)
            print(l, _minDCF)
        
        # BEWARE: kfold + balancing took me a lot to compute results, so they are commented in order to reduce waiting time!!!
        '''
        print("with balancing RAW")
        #with balancing
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "quad-b", norm=False)
            print(l, _minDCF)

        print("with balancing Normalized")
        #with balancing Normalized
        for l in [1e-6,1e-3,0.1,1.0]:
            _minDCF = k_fold_log_reg(DTR, LTR, 5, l, 0.5, 1, 1, "quad-b", norm=True)
            print(l, _minDCF)
        '''
    
    if SVM == True:
        #find C by plotting graphs:
        kfold_linsvm_wrapper(DTR,LTR, norm=False, gauss=False)
        kfold_linsvm_wrapper(DTR,LTR, norm=True, gauss=False)
        kfold_linsvm_wrapper(DTR,LTR, norm=True, gauss=True)

        kfold_quadsvm_wrapper(DTR, LTR, c = 0, gamma = 0, csi = 1, type="poly")
        kfold_quadsvm_wrapper(DTR, LTR, c = 1, gamma = 0, csi = 1, type="poly")

        kfold_quadsvm_wrapper(DTR, LTR, c = 0, gamma = 1, csi = 1, type="RBF")
        kfold_quadsvm_wrapper(DTR, LTR, c = 0, gamma = 2, csi = 1, type="RBF")

    if GMM == True:
        wrapper(DTR, LTR, covariance_type="Full")
        wrapper(DTR, LTR, covariance_type="Diagonal")
        wrapper(DTR, LTR, covariance_type="Tied")
        wrapper(DTR, LTR, covariance_type="Tied Diagonal")

def stats_and_preprocessing(DTR, LTR):
    dataset_test_stats(DTR,LTR)
    plot_histogram(DTR,LTR, title="Original Dataset")                #plot original dataset histograms

    # ****** Pre-processing ******
    plot_histogram(Z_normalization(DTR),LTR, title="Z-Normalized (centered)")         #plot centered (z-normalized) dataset histograms
    plot_histogram(Gaussianization(DTR,DTR),LTR, title="Gaussianized")                    #plot gaussianized dataset histograms
    plot_heatmap_pearson(DTR,LTR)                                    #plot Pearson coefficient heatmap to understand correlations

def actDCFVal(DTR, LTR):
    print(k_fold_log_reg_actDCF(DTR, LTR, 5, 1e-3, 0.5, 1, 1, "quad-ub", True))
    print(k_fold_quad_actDCF(DTR, LTR, 5, C = 0.1, c = 1, gamma = 2, csi = 1, type = "poly", norm = True))
    print(k_fold_quad_actDCF(DTR, LTR, 5, 1.0, 1, 1, 1, "RBF", norm=True))

if __name__ == '__main__':

    # ****** Load training from Train.txt ******
    D,L = load_data("data/Train.txt")
    TEST,L_TEST = load_data("data/Test.txt")

    #stats_and_preprocessing(D,L)

    if VALIDATION == True:
        single_fold(D, L)
        kfold(D, L, k = 5)
        actDCFVal(D, L)

    if EVALUATION == True:
        (DTR_60,LTR_60),(_,_) = split_db_2to1(D,L)
        print("60% TRAINING DATA")
        actDCFeval(DTR_60,LTR_60,TEST,L_TEST)
        #evaluation(DTR_60, LTR_60, TEST, L_TEST)    #testing with 66% of training data
        print("100% TRAINING DATA")
        #evaluation(D, L, TEST, L_TEST)    #testing with 100% of training data
        actDCFeval(D,L,TEST,L_TEST)

