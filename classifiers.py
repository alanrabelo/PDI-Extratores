import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

class Classifiers:

    def classify(self, humomments, labels, iterations=30, n_neighbors=5, title=''):

        classifiers = [
            KNeighborsClassifier(n_neighbors=n_neighbors),
            MLPClassifier(hidden_layer_sizes=(64, 128, 64),
                          learning_rate='adaptive',
                          learning_rate_init=0.005)
        ]

        scores_list = []

        for index, classifier in enumerate(classifiers):

            print(title + 'KNN' if index == 0 else 'MLP')
            for i in range(iterations):
                # k-Fold Cross-Validation
                # model = KNeighborsClassifier(n_neighbors=10)
                #

                # Cross-validation
                cv_scores = cross_val_score(classifier, humomments, labels, cv=5)
                scores = np.mean(cv_scores)
                scores_list.append(scores)

            acc = np.mean(scores_list)
            std = np.std(scores_list)

            print('Iterations: {:d}'.format(iterations))
            print('Neighbors: {:d}'.format(n_neighbors))
            print('Accuracy: {:.2f}'.format(acc * 100))
            print('Minimum: {:.2f}'.format(np.amin(scores_list) * 100))
            print('Maximum: {:.2f}'.format(np.amax(scores_list) * 100))
            print('Standard Deviation: {:.2f}\n\n'.format(std))
