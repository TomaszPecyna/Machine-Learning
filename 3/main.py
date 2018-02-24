import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from stemming.porter2 import stem
import re

def readFromCSV(names):
    return [np.array(pd.read_csv(name + ".csv", dtype='float64', header=None).as_matrix()) for name in names]


def plot(X, y):
    marker = {0: 'o', 1: '^'}
    color = {0: 'b', 1: 'r'}
    for x, y in zip(X, y):
        plt.scatter(x[0], x[1], marker=marker[y[0]], color = color[y[0]])

def plotGaussianBoundary(X, clf):
    h = 0.02
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

def predictionAccuracy(actualValues, predictions):
    return (predictions.shape[0] - np.sum(abs((actualValues - predictions)))) / predictions.shape[0]

def bestCoefficients(X, y, Xval, yval):
    accuracy = 0
    proposedValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigmaTest in proposedValues:
        for CTest in  proposedValues:
            sigmaToGamma = lambda x: 1 / (x ** 2 * 2)
            clf = svm.SVC(kernel='rbf', gamma=sigmaToGamma(sigmaTest), C=CTest)
            clf.fit(X, y.flatten())
            predictions = clf.predict(Xval)
            accuracyTMP = predictionAccuracy(yval.flatten(), predictions)
            if(accuracyTMP > accuracy):
                accuracy = accuracyTMP
                sigma = sigmaTest
                C = CTest

    return C, sigma

def processEmail(data):
    # some text formatting #
    data = data.lower()
    data = re.sub(r'<[^<>]+>', r' ', data)
    data = re.sub(r'[0-9]+', 'number', data)
    data = re.sub(r'(http|https)://[^\s]*', 'httpaddr', data)
    data = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', data)
    data = re.sub(r'[$]+', 'dollar', data)
    data = re.sub('[^a-zA-Z0-9]', ' ', data)
    data = data.split()

    data = [stem(word) for word in data]
    return data



def main():
    names = ["X1", "X2", "X3", "XspamTest", "XspamTrain", "y1", "y2", "y3", "yspamTest", "yspamTrain", "X3val", "y3val"]
    X1, X2, X3, XspamTest, XspamTrain, y1, y2, y3, yspamTest, yspamTrain, X3val, y3val = readFromCSV(names)
    sigmaToGamma = lambda x: 1/(x**2 * 2)

    # playing with linear kernel #

    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(X1, y1.flatten())

    coef = clf.coef_[0]
    a = -coef[0] / coef[1]
    xx = np.linspace(0, 4)
    yy = a * xx - (clf.intercept_[0]) / coef[1]
    plt.plot(xx, yy, 'k-')
    plot(X1, y1)
    plt.show()

    # playing with gaussian kernel #

    clf = svm.SVC(kernel='rbf', gamma=sigmaToGamma(0.1), C=1)
    clf.fit(X2, y2.flatten())
    plotGaussianBoundary(X2, clf)
    plot(X2, y2)
    plt.show()

    # more playing with gaussian kernel, testing different parameters #

    C, sigma = bestCoefficients(X3, y3, X3val, y3val)
    clf = svm.SVC(kernel='rbf', gamma=sigmaToGamma(sigma), C=C)
    clf.fit(X3, y3.flatten())
    plotGaussianBoundary(X3, clf)
    plot(X3, y3)
    plt.show()

    # spam classification #

    with open('emailSample1.txt', 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
    with open('vocab.txt', 'r') as myfile:
            vocab = myfile.read().replace('\n', ' ')
    vocab = vocab.split()
    vocab = {vocab[x+1] : (int(vocab[x]) - 1) for x in range(0, len(vocab),2)}
    data = processEmail(data)
    data = [vocab[word] for word in data if word in vocab]
    vector = np.zeros((len(vocab), 1))
    for word in data:
        vector[word] = 1

    C, sigma = bestCoefficients(XspamTrain, yspamTrain, XspamTest, yspamTest)
    clf = svm.SVC(kernel='rbf', gamma=sigmaToGamma(sigma), C=C)
    clf.fit(XspamTrain, yspamTrain.flatten())
    trainAccuracy = predictionAccuracy(yspamTrain.flatten(), clf.predict(XspamTrain))
    testAccuracy = predictionAccuracy(yspamTest.flatten(), clf.predict(XspamTest))
    print(trainAccuracy, testAccuracy, C, sigma)

if __name__ == "__main__":
    main()