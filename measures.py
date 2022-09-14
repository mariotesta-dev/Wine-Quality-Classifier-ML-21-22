"""
*** PREDICTIONS MEASUREMENT ***
Functions to analyse model performances and do some plots. In particular:
* calculate from the log-likelihood ratios the minimum DCF and the actual DCF using the theoretical threshold
* print ROC curves and Bayes error plots from the log-likelihood ratios
* calculate Bayes risk, optimal Bayes decisions and confusion matrixes
"""

import numpy

def confusion_matrix(predicted_labels, LTE):

    # extract the different classes
    classes = numpy.unique(LTE)

    # initialize the confusion matrix
    confmat = numpy.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[j, i] = numpy.sum((LTE == classes[i]) & (predicted_labels == classes[j]))

    return confmat

# Calculate unnormalized DCF and normalized DCF
def bayes_risk(M, pi, Cfn, Cfp):
    
    FNR = M[0,1] / (M[0,1] + M[1,1])
    FPR = M[1,0] / (M[0,0] + M[1,0])

    DCFu = pi * Cfn * FNR + (1 - pi) * Cfp * FPR
    B_dummy = min(pi * Cfn, (1 - pi) * Cfp)
    DCF = DCFu / B_dummy

    return DCFu, DCF

# Calculate data to print the ROC plot (FPR and TPR for different thresholds)
def ROC_curve(llr, true_labels):

    possible_t = numpy.concatenate((numpy.array([min(llr) - 0.1]), (numpy.unique(llr)), numpy.array([max(llr) + 0.1])))
    FPR = numpy.zeros([possible_t.shape[0]])
    TPR = numpy.zeros([possible_t.shape[0]])

    for index, t in enumerate(possible_t):

        PredictedLabels = numpy.zeros([llr.shape[0]])
        PredictedLabels[llr > t] = 1
        M = confusion_matrix(true_labels, PredictedLabels)
        FNR = M[0,1] / (M[0,1] + M[1,1])
        FPR[index] = M[1,0] / (M[0,0] + M[1,0])
        TPR[index] = 1 - FNR

    return FPR, TPR

# Compute llr from class conditional log probabilities (just subtract the
# values for the two classes)
def compute_llr(s):
    if s.shape[0] != 2:
        return 0
    return s[1, :] - s[0, :]

##### CORRECT ONES #####

def optimal_bayes_decisions(llr, pi1, Cfn, Cfp, threshold=None):
    """ Computes optimal Bayes decisions starting from the binary 
        log-likelihoods ratios
        llr is the array of log-likelihoods ratios
        pi1 is the prior class probability of class 1 (True)
        Cfp = C1,0 is the cost of false positive errors, that is the cost of 
        predicting class 1 (True) when the actual class is 0 (False)
        Cfn = C0,1 is the cost of false negative errors that is the cost of 
        predicting class 0 (False) when the actual class is 1 (True)
    """

    # initialize an empty array for predictions of samples
    predictions = numpy.empty(llr.shape, int)
    threshold = - numpy.log((pi1 * Cfn) / ((1 - pi1) * Cfp))

    for i in range(llr.size):
        if llr[i] > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions

def empirical_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
    """ Computes the Bayes risk (or detection cost) from the consufion matrix 
        corresponding to the optimal decisions for an 
        application (pi1, Cfn, Cfp)
    """

    # FNR = false negative rate
    FNR = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])

    # FPR = false positive rate
    FPR = confusion_matrix[1][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

    # We can compute the empirical bayes risk, or detection cost function DCF
    # using this formula
    DCF = pi1 * Cfn * FNR + (1-pi1) * Cfp * FPR

    return DCF

def normalized_detection_cost(DCF, pi1, Cfn, Cfp):
    """ Computes the normalized detection cost, given the detection cost DCF,
        and the parameters of the application, pi1, Cfn, Cfp
    """

    # We can compute the normalized detection cost (or bayes risk)
    # by dividing the bayes risk by the risk of an optimal system that doen not
    # use the test data at all

    # The cost of such system is given by this formula
    DCFdummy = pi1 * Cfn if (pi1 * Cfn < (1-pi1) * Cfp) else (1-pi1) * Cfp

    return DCF / DCFdummy

def compute_predicted_labels(llr, t):

    PredictedLabel = []

    for i in llr:
        if i > t:
            PredictedLabel.append(1)
        else:
            PredictedLabel.append(0)

    return numpy.array(PredictedLabel)


def minimum_detection_cost(llr, actual_labels, pi, Cfn, Cfp):
    """ Compute the minimum detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    # 1) ordina in ordine crescente i test scores= data (logLikelihood ratios)
    thresholds = numpy.append(llr, [numpy.inf, -numpy.inf])
    thresholds.sort()

    # 2) considero ogni elemento data come threshold, ottengo le predicted labels confrontando con la threshold
    
    _minDCF = numpy.inf
    for t in thresholds:
        predicted_labels = compute_predicted_labels(llr,t)
        conf_matrix = confusion_matrix(predicted_labels, actual_labels)
        DCF = empirical_bayes_risk(conf_matrix, pi, Cfn, Cfp)
        norm_DCF = normalized_detection_cost(DCF, pi, Cfn, Cfp)
        
        if norm_DCF < _minDCF:
            _minDCF = norm_DCF

    return _minDCF

def actual_detection_cost(llr, labels, pi1, Cfn, Cfp):
    """ Compute the actual detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    # compare the log-likelihood ratio with  the optimal bayes decision threshold to predict the class
    predictions = optimal_bayes_decisions(llr, pi1, Cfn, Cfp)

    # compute the confusion matrix
    conf = confusion_matrix(predictions, labels)

    # compute DCF_norm
    DCF = empirical_bayes_risk(conf, pi1, Cfn, Cfp)
    DCF_norm = normalized_detection_cost(DCF, pi1, Cfn, Cfp)

    return DCF_norm

def t_calibrated(llr, actual_labels, pi, Cfn, Cfp):
    """ Compute the minimum detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    # 1) ordina in ordine crescente i test scores= data (logLikelihood ratios)
    thresholds = numpy.append(llr, [numpy.inf, -numpy.inf])
    thresholds.sort()

    # 2) considero ogni elemento data come threshold, ottengo le predicted labels confrontando con la threshold
    optT = None
    _minDCF = numpy.inf
    for t in thresholds:
        predicted_labels = compute_predicted_labels(llr,t)
        conf_matrix = confusion_matrix(predicted_labels, actual_labels)
        DCF = empirical_bayes_risk(conf_matrix, pi, Cfn, Cfp)
        norm_DCF = normalized_detection_cost(DCF, pi, Cfn, Cfp)
        
        if norm_DCF < _minDCF:
            optT = t
            _minDCF = norm_DCF

    return optT

def minDCF_given_t(llr, actual_labels, pi, Cfn, Cfp, t):

    predicted_labels = compute_predicted_labels(llr,t)
    conf_matrix = confusion_matrix(predicted_labels, actual_labels)
    DCF = empirical_bayes_risk(conf_matrix, pi, Cfn, Cfp)
    norm_DCF = normalized_detection_cost(DCF, pi, Cfn, Cfp)

    return norm_DCF