import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import optimize

def linearRegCostFunction(theta, X, y, penalty):
    difference = np.sum(np.multiply(X, theta), axis=1).reshape(y.shape[0], 1) - y
    J = sum(1 / (2 * y.shape[0]) * difference ** 2) + penalty / (2 * y.shape[0]) * sum(theta[1:] ** 2)
    grad = [1 / y.shape[0] * sum(difference * X[:, j].reshape(y.shape[0], 1)) if j == 0
            else 1 / y.shape[0] * sum(difference * X[:, j].reshape(y.shape[0], 1)) + penalty / y.shape[0] * theta[j]
           for j in range(theta.shape[0])]
    return [J, (np.asarray(grad)).flatten()]

def readFromCSV(names):
    return [np.array(pd.read_csv(name + ".csv", dtype='float64', header=None).as_matrix()) for name in names]

def trainLinearReg(X, y, penalty):
    initialTheta = np.zeros((X.shape[1]))
    args = (X, y, penalty)
    J = lambda theta, X, y, penalty : linearRegCostFunction(theta, X, y, penalty)[0]
    grad = lambda theta, X, y, penalty: linearRegCostFunction(theta, X, y, penalty)[1]
    return sp.optimize.fmin_cg(J, initialTheta, grad, args=args, maxiter=200, disp=0)

def learningCurve( X, y, Xval, yval, penalty):
    errorTrain, errorVal = [], []
    for x in range(X.shape[1], y.shape[0] + 1):
        theta = trainLinearReg(np.asarray(X[0:x]), y[0:x], penalty)
        errorTrain.append(linearRegCostFunction(theta, X[0:x], y[0:x], 0)[0])
        errorVal.append(linearRegCostFunction(theta, Xval, yval, 0)[0])
    return (errorTrain, errorVal)

def polyFunction(X, p):
    p += 1
    return (np.ones((X.shape[0], p)) * X) ** np.arange(p)

def normalize(X, avg = None, std = None):
    returnValue = []
    if(avg is None):
        avg = np.average(X, axis=0)
        returnValue.append(avg)
    if(std is None):
        std = np.std(X, axis=0)
        returnValue.append(std)
    returnValue.append((X - avg) / std)
    returnValue[-1] = np.nan_to_num(returnValue[-1])
    returnValue[-1][:, 0] += 1
    return returnValue

def main():
    names = ["X", "y", "Xval", "yval", "Xtest", "ytest"]
    X, y, Xval, yval, Xtest, ytest = readFromCSV(names)

    #checking whether cost function works
    penalty = 1
    theta = np.array([1, 1])
    J, grad = linearRegCostFunction(theta, np.concatenate([np.ones((X.shape[0], 1)), X], axis=1), y, penalty)

    print(J, grad)

    #showing y = ax + b
    penalty = 1
    theta = trainLinearReg(np.concatenate([np.ones((X.shape[0], 1)), X], axis=1), y, penalty)
    xes = np.arange(-1.5, 2, 0.1)
    xes = xes.reshape(xes.shape[0], 1)
    function = lambda xes, theta: np.sum(np.multiply(xes, theta), axis=1)
    plt.plot(X, y, 'o')
    plt.plot(xes, function(polyFunction(xes, 1), theta), '-')
    plt.show()

    print(theta)

    #trainerror with y = ax + b
    errorTrain, errorVal = learningCurve(np.concatenate([np.ones((X.shape[0], 1)), X], axis=1), y,
                                         np.concatenate([np.ones((Xval.shape[0], 1)), Xval], axis=1), yval, penalty)

    plt.plot(np.arange(2, y.shape[0] + 1), errorTrain, '-')
    plt.plot(np.arange(2, y.shape[0] + 1), errorVal, '-')
    plt.show()

    #adding more features (x^2, x^3... x^p)
    p = 8
    XPoly = polyFunction(X, p)
    avg, std, XPoly = normalize(XPoly)

    XPolyTest = polyFunction(Xtest, p)
    XPolyTest = normalize(XPolyTest, avg, std)

    XPolyVal = polyFunction(Xval, p)
    XPolyVal = normalize(XPolyVal, avg, std)

    theta = trainLinearReg(XPoly, y, penalty)

    plt.plot(XPoly[:, 1], y, 'o')
    plt.plot(xes, function(polyFunction(xes, p), theta), '-')
    plt.show()



if __name__ == "__main__":
    main()
