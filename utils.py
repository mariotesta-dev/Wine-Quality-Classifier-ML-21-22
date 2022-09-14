from numpy.lib.function_base import corrcoef
import numpy
import matplotlib.pyplot as plt

from measures import minimum_detection_cost

#Generics
def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def load_data(filename):

    f = open(filename, "r")
    values = []
    labels = []

    for line in f:
        line = line.rstrip().split(",")
        label = line[11]
        entry = numpy.array(line[0:11], dtype=numpy.float32).reshape(11,1)

        values.append(entry)
        labels.append(label)
    
    return numpy.hstack(values), numpy.array(labels, dtype=numpy.int32)

def plot_2D_data(D, L, folder=None, name=None):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    plt.figure()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.scatter(D0[0], D0[1], label='Bad-Quality')
    plt.scatter(D1[0], D1[1], label='Good-Quality')

    plt.legend()
    plt.tight_layout()
    #plt.savefig(folder+'/'+ name +'.pdf')
    plt.show()

def plot_histogram(values, labels, folder=None, name=None, title=None):
    v0 = values[:, labels == 0]
    v1 = values[:, labels == 1]

    features = {
        0: 'Fixed Acidity',
        1: 'Volatile Acidity',
        2: 'Citric Acid',
        3: 'Residual Sugar',
        4: 'Chlorides',
        5: 'Free Sulfur Dioxide',
        6: 'Total Sulfur Dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol',
    }

    for index in range(values.shape[0]):
        plt.figure()
        if title != None:
            plt.title(title)
        plt.xlabel(features[index])
        plt.hist(v0[index, :], bins = 10, density = True, alpha = 0.4, label = 'Bad-Quality')
        plt.hist(v1[index, :], bins = 10, density = True, alpha = 0.4, label = 'Good-Quality')

        plt.legend()
        #plt.savefig('%s/%s_%d.pdf' % (folder,name,index))
    plt.show()

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def kfold(D, L, k):
    data_folds = numpy.array_split(D, k, axis=1)
    label_folds = numpy.array_split(L, k, axis=0)

    d_train = []
    d_test = []

    cross_val_data = {'train': d_train, 'test':d_test}
    for i, test_i in enumerate(data_folds):
        d_train.append(data_folds[:i] + data_folds[i+1:])
        d_test.append(test_i)
    
    l_train = []
    l_test = []

    cross_val_label = {'train': l_train, 'test':l_test}
    for i, test_i in enumerate(label_folds):
        l_train.append(label_folds[:i] + label_folds[i+1:])
        l_test.append(test_i)
    return cross_val_data, cross_val_label


def plot_heatmap_pearson(D, L, name=None):

    features = [
        'Fixed Acidity',
        'Volatile Acidity',
        'Citric Acid',
        'Residual Sugar',
        'Chlorides',
        'Free Sulfur Dioxide',
        'Total Sulfur Dioxide',
        'Density',
        'pH',
        'Sulphates',
        'Alcohol',
        'Quality'
    ]

    print("**** Plotting Correlation Matrix... ****")
    
    pearson_matrix = corrcoef(numpy.vstack([D,L]))

    fig, ax = plt.subplots()
    im = ax.imshow(pearson_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(numpy.arange(len(features)), labels=features, fontsize='xx-small')
    ax.set_yticks(numpy.arange(len(features)), labels=features, fontsize='xx-small')

    # Rotate the tick labels and set their alignment.   
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax.text(j, i, '{0:.2f}'.format(pearson_matrix[i, j]),ha="center", va="center", color="w")

    #plt.savefig('./results/%s.pdf' % (name))
    plt.show()
    print("**** Plotting Ended... ****\n")
    return pearson_matrix

def dataset_train_stats(D,L):

    #Count how many samples per classes
    c0 = len(L[L == 0])
    c1 = len(L[L == 1])

    print("WINE QUALITY TRAINING SET")
    print("Total number of samples = %d" % (c0+c1))
    print("L0 = %d samples, L1 = %d samples" % (c0, c1))

    print("\n**** Plotting Histograms... ****")
    #plot_histogram(D,L,folder="results/hist",name="feature")
    print("**** Plotting Ended... ****\n")

    return c0,c1

def dataset_test_stats(D,L):

    #Count how many samples per classes
    c0 = len(L[L == 0])
    c1 = len(L[L == 1])

    print("WINE QUALITY TEST SET")
    print("Total number of samples = %d" % (c0+c1))
    print("L0 = %d samples, L1 = %d samples" % (c0, c1))

    return c0,c1

def kfold(D, L, k):

    data_folds = numpy.array_split(D, k, axis=1)
    label_folds = numpy.array_split(L, k, axis=0)

    d_train = []
    d_test = []

    cross_val_data = {'train': d_train, 'test':d_test}
    for i, test_i in enumerate(data_folds):
        d_train.append(data_folds[:i] + data_folds[i+1:])
        d_test.append(test_i)
    
    l_train = []
    l_test = []

    cross_val_label = {'train': l_train, 'test':l_test}
    for i, test_i in enumerate(label_folds):
        l_train.append(label_folds[:i] + label_folds[i+1:])
        l_test.append(test_i)
    return cross_val_data, cross_val_label


def k_fold_new(D, L, K, classificator, params, seed=0):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    numpy.random.seed(seed)

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

        llr = classificator(DTR, LTR, DTE, params)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    minDCF = minimum_detection_cost(all_llr, all_labels, 0.5, 1, 1)
    
    return minDCF
