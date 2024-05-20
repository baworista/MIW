import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict_proba(self, X):
        return self.activation(self.net_input(X))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class SoftmaxRegression(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            lr = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            lr.fit(X, y_binary)
            self.classifiers[c] = lr

    def predict(self, X):
        predictions = []
        for x in X:
            probs = np.zeros(len(self.classes))
            for c, clf in self.classifiers.items():
                score = clf.net_input(x)
                probs[c] = clf.activation(score)
            predicted_class = np.argmax(probs)
            predictions.append(predicted_class)
        return np.array(predictions)



def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    softmax_regressor = SoftmaxRegression(eta=0.01, n_iter=100000, random_state=1)
    softmax_regressor.fit(X_train, y_train)

    # Sprawdzenie dokładności na zestawie testowym
    y_pred = softmax_regressor.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    plot_decision_regions(X=X_train, y=y_train, classifier=softmax_regressor)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
