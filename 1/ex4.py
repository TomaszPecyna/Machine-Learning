import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import log, exp, sqrt
import csv

########################################################################################################################

numberOfTrainingExamples = 5000
inputLayerSize = 400
hiddenLayerSize = 25
numberOfLabels = 10
numberOfLayers = 3

########################################################################################################################

def costFunction(weights, inputLayyerSize, hiddenLayerSize, numberOfLabels, data, actualValues, g, regularizationParameter = 0):
    identity = np.identity(10)
    cost = 0
    for exampleNumber in range(data.shape[0]):

        outputLayer = networkOutput(data, exampleNumber, weights, g)

        for actualOutputValue, neuralOutputValue in zip(identity[int(actualValues[exampleNumber] - 1)], outputLayer):
            cost =  cost - actualOutputValue * log(neuralOutputValue) - (1 - actualOutputValue) * log(1 - neuralOutputValue)
    cost /= data.shape[0]
    regularization = 0
    for weightOnLayer in weights:
        regularization += np.sum(np.square(weightOnLayer[:, 1:]))
    regularization = (regularizationParameter * regularization) / (2 * data.shape[0])
    return cost + regularization

#returns activation or multiplication matrix from a given layer, layers between 2 and numberOfLayers and -1, network output default
def networkOutput(data, example, weights, g, which = 'activations', layer = -1):

    multiplications = []
    activations = []
    if layer != -1:
        layer -= 2

    multiplications.append(weights[0].dot((np.append([1], data[example])).transpose()))
    activations.append(np.array([g(x) for x in multiplications[0]]))

    for number in range(numberOfLayers - 2):
        multiplications.append(weights[number + 1].dot(np.append([1], activations[number])))
        activations.append(np.array([g(x) for x in multiplications[number + 1]]))

    returnValue = -1
    if which == 'activations':
        returnValue = activations[layer]
    if which == 'multiplications':
        returnValue = multiplications[layer]
    return returnValue

def gradientChecking(aaa, derivatives, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues, g, regularizationParameter = 0):

    for weightRow in range(aaa[0].shape[0]):
        for a in range(aaa[0].shape[1]):
            littleMore = []
            littleMore.append(np.ones(aaa[0].shape) * aaa[0])
            littleMore.append(np.ones(aaa[1].shape) * aaa[1])
            littleMore[0][weightRow][a] += 0.0001

            littleLess = []
            littleLess.append(np.ones(aaa[0].shape) * aaa[0])
            littleLess.append(np.ones(aaa[1].shape) * aaa[1])
            littleLess[0][weightRow][a] -= 0.0001

            moreCost = costFunction(littleMore, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues,
                                    g, regularizationParameter)
            #print(moreCost)
            lessCost = costFunction(littleLess, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues,
                                    g, regularizationParameter)
            #print(lessCost)
            calculatedDerivative = (moreCost - lessCost) / (2 * 0.0001)

            print(calculatedDerivative, derivatives[0][weightRow][a])

def calculatingDerivatices(randomWeights, sigmoidGradient):

    errorTerms = []
    errorTerms.append(np.zeros(numberOfLabels))
    errorTerms.insert(0, np.zeros(hiddenLayerSize + 1))

    accumulatedErrors = []
    accumulatedErrors.append(np.zeros((hiddenLayerSize, inputLayerSize + 1)))
    accumulatedErrors.append(np.zeros((numberOfLabels, hiddenLayerSize + 1)))

    identity = np.identity(10)
    regularizationParameter = 1

    for exampleNumber in range(data.shape[0]):
        errorTerms[-1] = networkOutput(data, exampleNumber, randomWeights, g) - identity[
            int(actualValues[exampleNumber] - 1)]
        errorTerms[-2] = (randomWeights[1].transpose()).dot(errorTerms[-1]) * np.append([1],
                                                                                        [sigmoidGradient(x) for x in
                                                                                         networkOutput(data,
                                                                                                       exampleNumber,
                                                                                                       randomWeights, g,
                                                                                                       'multiplications',
                                                                                                       2)])

        accumulatedErrors[-1] += errorTerms[-1].reshape(10, 1).dot(
            np.append([1], networkOutput(data, exampleNumber, randomWeights, g, 'activations', 2)).reshape(1,
                                                                                                           hiddenLayerSize + 1))
        accumulatedErrors[-2] += errorTerms[-2][1:].reshape(25, 1).dot(
            np.append([1], data[exampleNumber]).reshape(1, inputLayerSize + 1))
    finalDerivatives = []

    matrixhehe = np.append(np.zeros((randomWeights[0].shape[0], 1)), randomWeights[0][:, 1:])

    finalDerivatives.append((np.ones((hiddenLayerSize, inputLayerSize + 1)) * accumulatedErrors[0]
                             + regularizationParameter * np.append(np.zeros((randomWeights[0].shape[0], 1)),
                                                                   randomWeights[0][:, 1:],
                                                                   axis=1)) / numberOfTrainingExamples)
    finalDerivatives.append((np.ones((numberOfLabels, hiddenLayerSize + 1)) * accumulatedErrors[1]
                             + regularizationParameter * np.append(np.zeros((randomWeights[1].shape[0], 1)),
                                                                   randomWeights[1][:, 1:],
                                                                   axis=1)) / numberOfTrainingExamples)

    #gradientChecking(randomWeights, finalDerivatives, inputLayerSize, hiddenLayerSize, numberOfLabels, data,
     #                actualValues, g, regularizationParameter)
    return finalDerivatives

########################################################################################################################

#reading all data needed
pandasData = pd.read_csv('data.csv', dtype='float64', header = None)
pandasActualValues = pd.read_csv('actualValues.csv', dtype='float64', header = None)
pandasTheta1 = pd.read_csv('Theta1.csv', dtype='float64', header = None)
pandasTheta2 = pd.read_csv('Theta2.csv', dtype='float64', header = None)

#pandas into numpy, yo
data = np.zeros(pandasData.shape)
actualValues = pandasActualValues.as_matrix().flatten()
weights = [np.array(pandasTheta1.as_matrix())]
weights.append(np.array(pandasTheta2.as_matrix()))
for row, i in zip(pandasData.values, range(numberOfTrainingExamples)):
    data[i, :] = np.add(data[i, :], row)

g = lambda x: 1 / (1 + exp(-x))
sigmoidGradient = lambda x: g(x) * (1 - g(x))
alpha = 1

cost = costFunction(weights, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues, g, 1)

print(cost)


epsylonInit1 = sqrt(6) / sqrt(inputLayerSize + hiddenLayerSize)
epsylonInit2 = sqrt(6) / sqrt(hiddenLayerSize + numberOfLabels)

randomWeights = []
randomWeights.append(np.random.rand(hiddenLayerSize, inputLayerSize + 1) * epsylonInit1 * 2 - epsylonInit1)
randomWeights.append(np.random.rand(numberOfLabels, hiddenLayerSize +1) * epsylonInit2 * 2 - epsylonInit2)


print(costFunction(randomWeights, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues, g, 1))

for i in range(10000000000000):
    derivatives = calculatingDerivatices(randomWeights, sigmoidGradient)
    randomWeights[0] -= alpha*derivatives[0]
    randomWeights[1] -= alpha * derivatives[1]
    print(costFunction(randomWeights, inputLayerSize, hiddenLayerSize, numberOfLabels, data, actualValues, g, 1))

    np.savetxt('weights0.csv', randomWeights[0], delimiter=',')
    np.savetxt('weights1.csv', randomWeights[1], delimiter=',')
