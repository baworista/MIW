import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class TwoClassPerceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.perceptrons = []

    def fit(self, X, y):
        unique_classes = np.unique(y)
        unique_classes = unique_classes[:-1]
        print(unique_classes)
        for class_label in unique_classes:
            print("Iteration ", class_label)
            perceptron = Perceptron(eta=self.eta, n_iter=self.n_iter)
            y_binary = np.where(y == class_label, 1, -1)
            perceptron.fit(X, y_binary)
            self.perceptrons.append(perceptron)
            X = X[y != class_label]
            y = y[y != class_label]

    def predict(self, X):
        print(self.perceptrons.__len__())
        predictions = []
        for perceptron in self.perceptrons:
            predictions.append(perceptron.predict(X))
        predictions = np.array(predictions)
        final_predictions = np.where(np.all(predictions == -1, axis=0), 2, np.argmax(predictions, axis=0))
        return final_predictions


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    two_class_perceptron = TwoClassPerceptron(eta=0.1, n_iter=1000)
    two_class_perceptron.fit(X_train, y_train)

    # Sprawdzenie dokładności na zestawie testowym
    y_pred = two_class_perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    plot_decision_regions(X=X_train, y=y_train, classifier=two_class_perceptron)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
