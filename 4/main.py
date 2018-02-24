import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
from sklearn import preprocessing

def readFromCSV(names):
    return [np.array(pd.read_csv(name + ".csv", dtype='float64', header=None).as_matrix()) for name in names]

def findClosestCentroids(X, initialCentroids):
    return [np.argmin([np.linalg.norm(np.subtract(centroid, x)) for centroid in initialCentroids]) for x in X]

def computeCentroids(X, indices, K):
    centroids = np.zeros((K, X.shape[1]))
    numberOfAssignments = np.zeros(K).reshape((1, K))
    for x, idx in zip(X, indices):
        centroids[idx] += x
        numberOfAssignments[0, idx] += 1
    numberOfAssignments = np.repeat(numberOfAssignments, X.shape[1], axis=0).transpose()
    return np.divide(centroids, numberOfAssignments)

def plot(X, y):
    color = {0: 'b', 1: 'r', 2: 'g'}
    for x, y in zip(X, y):
        plt.scatter(x[0], x[1], color = color[y], zorder=1)

def kMeans(X, centroids, K, maxIters, draw=True):
    indices = findClosestCentroids(X, centroids)
    for i in range(maxIters):
        previousCentroids = centroids
        if(draw):
            for centroid in centroids:
                plt.scatter(centroid[0], centroid[1], color='k', marker='x', zorder=2, s=100)
        indices = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, indices, K)
        if(np.array_equal(previousCentroids, centroids)):
            break
        if(draw):
            for pairs in zip(zip(previousCentroids, centroids)):
                a = np.transpose(np.asarray(pairs)).flatten()
                xses, ys = a.reshape((2, int(a.size/2)))
                plt.plot(xses, ys, 'k')
    if(draw):
        plot(X, indices)
        plt.show()
    return centroids

def reshapeData(X):
    return X.reshape((int(X.size/X.shape[-1]), X.shape[-1]))

def kMeansInitCentroids(data, k):
    randomIndices = random.sample(list(range(0, data.shape[0])), k)
    return np.asarray([data[randomIndex] for randomIndex in randomIndices])



def main():
    names = ["ex7data1", "ex7data2", "ex7faces"]
    data1, data2, faces = readFromCSV(names)

    # playing with k means algorithm #
    k = 3
    maxIters = 10
    initialCentroids = kMeansInitCentroids(data2, k)
    print(kMeans(data2, initialCentroids, k, maxIters))

    # image compression with k-means #

    image = cv2.imread('bird_small.png')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    k = 16
    maxIters = 100
    reshapedImage = reshapeData(image)
    centroids = kMeans(reshapedImage, kMeansInitCentroids(reshapedImage, k), k, maxIters, draw=False)
    indices = findClosestCentroids(reshapedImage, centroids)
    bufferedImage = np.asarray([np.uint8(centroids[index]) for index in indices])
    bufferedImage = bufferedImage.reshape(image.shape)
    plt.imshow(cv2.cvtColor(bufferedImage, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()