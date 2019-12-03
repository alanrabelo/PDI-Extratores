import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

class Classifiers:

    def classify(self, humomments, lb, labels, iterations=30, n_neighbors=5, title=''):

        classifiers = [
            KNeighborsClassifier(n_neighbors=n_neighbors),
            MLPClassifier(hidden_layer_sizes=(64, 128, 64),
                          max_iter=1000,
                          learning_rate_init=0.001)
        ]

        scores_list = []

        for index, classifier in enumerate(classifiers):

            for i in range(iterations):

                print(title + 'KNN' if index == 0 else 'MLP')

                x_train, x_test, y_train, y_test = train_test_split(humomments, labels, test_size=0.2)
                model = classifier
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                if i == 0:
                    # Confusion matrix
                    y_pred_inv = lb.inverse_transform(y_pred)
                    y_test_inv = lb.inverse_transform(y_test)
                    cm = confusion_matrix(y_pred_inv, y_test_inv)
                    print(cm)

                scores = accuracy_score(y_pred, y_test)
                scores_list.append(scores)

            acc = np.mean(scores_list)
            std = np.std(scores_list)

            print('Número de iterações: %d' % iterations)
            print('Número de vizinhos: %d' % n_neighbors)
            print('Accuracy: %.2f' % (acc * 100))
            print('Minimum: %.2f' % (np.amin(scores_list) * 100))
            print('Maximum: %.2f' % (np.amax(scores_list) * 100))
            print('Standard Deviation: %.2f\n\n' % std)
