import numpy
import matplotlib.pyplot as plt
from utils import vcol, vrow

'''
    This module contains all function written during the course
    and helpful for the final project.
'''


def plot_histogram(values, labels, folder=None):
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
        plt.xlabel(features[index])
        plt.hist(v0[index, :], bins = 10, density = True, alpha = 0.4, label = 'Bad-Quality')
        plt.hist(v1[index, :], bins = 10, density = True, alpha = 0.4, label = 'Good-Quality')

        plt.legend()
        #plt.savefig(folder+'/histogram_%d.pdf' % index)
    plt.show()

def plot_scatter(values, labels, folder=None):
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

    for index_1 in range(values.shape[0]):
        for index_2 in range(values.shape[0]):
            if index_1 == index_2:
                continue
            else:
                plt.figure()
                plt.xlabel(features[index_1])
                plt.ylabel(features[index_2])
                plt.scatter(v0[index_1, :], v0[index_2, :], label = 'Bad-Quality')
                plt.scatter(v1[index_1, :], v1[index_2, :], label = 'Good-Quality')

                plt.legend()
                #plt.savefig(folder+'/scatter_%d_%d.pdf' % (index_1, index_2))
        plt.show()













