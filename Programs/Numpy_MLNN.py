# Written in Python 3.8.1
# Version specific features (e.g. "final" syntax) may preclude execution on earlier versions

# Kobe Davis
# Prof. Doliotis
# CS 445
# 10 February 2020
#
# Assignment 2: Neural Networks

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import final

np.set_printoptions(threshold=np.inf, linewidth=200)

# Important Constants
NUMBER_OF_CLASSES: final = 10   # Number of rows in weight matrix
NUMBER_OF_EPOCHS: final = 50
NUMBER_OF_HIDDEN: final = 40   # 20, 50, 100
EXAMPLES_PERCENT: final = 1  # 1 -> All, 0.5 -> Half, 0.25 -> Quarter, etc.
MAX_ACCURACY: final = 100
LEARNING_RATE: final = 0.001      # 0.1, 0.01, 0.001
TRAINING_DATA: final = "mnist_train.csv"
TESTING_DATA: final = "mnist_test.csv"


def main():
    startTime = time.time()
    #print("\nConfirm that the following parameters are correct.\n")
    #print(f'Number of classifications: {NUMBER_OF_CLASSES}')
    #print(f'Number of epochs: {NUMBER_OF_EPOCHS}')
    #print(f'Number of hidden units: {NUMBER_OF_HIDDEN}')
    #print(f'Proportion of training examples used: {EXAMPLES_PERCENT}')
    #print(f'Learning rate: {LEARNING_RATE}')
    #print(f'Training file: {TRAINING_DATA}')
    #print(f'Test file: {TESTING_DATA}')
    #input("Press ENTER to continue.\n")

    print("\nLoading data from files...")
    inTrain, inTest, ih_WMAT, ho_WMAT, szTrain, szTest = load()
    print(f'\tNumber of training examples: {szTrain}')
    print(f'\tNumber of test examples: {szTest}')

    print("Preprocessing...")
    gtTrain, gtTest, inTrain, inTest = preprocess(
        inTrain, szTrain, inTest, szTest)


    print("Beginning epochs...\n")
    accTrain = np.zeros(NUMBER_OF_EPOCHS)
    accTest = np.zeros(NUMBER_OF_EPOCHS)
    for i in range(NUMBER_OF_EPOCHS):
        print(f'Epoch {i}')
        print("\tTraining Network...")
        ih_WMAT, ho_WMAT = epoch(
            inTrain, szTrain, ih_WMAT, ho_WMAT, gtTrain)

        print("\tComputing Accuracy...")
        accTrain[i] = reportAccuracy(
            inTrain, szTrain, ih_WMAT, ho_WMAT, gtTrain, 0)
        #if i == NUMBER_OF_EPOCHS-1:
        #    print("\t\tGenerating confusion matrix...")
        #    accTest[i], confusion = reportAccuracy(
        #        inTest, szTest, ih_WMAT, ho_WMAT, gtTest, 1)
        #else:
        accTest[i] = reportAccuracy(
            inTest, szTest, ih_WMAT, ho_WMAT, gtTest, 0)
        print(
            f'\t\tTraining accuracy: {accTrain[i]}/{szTrain} = {(accTrain[i]/szTrain)*100:.2f}%')
        print(
            f'\t\tTest accuracy: {accTest[i]}/{szTest} = {(accTest[i]/szTest)*100:.2f}%\n')

    endTime = time.time()
    print(f'Time taken: {endTime-startTime} seconds')

    #print(f'Confusion Matrix:\n{confusion}')
    #percAccTrain = 100*(accTrain/szTrain)
    #percAccTest = 100*(accTest/szTest)
    #plotAccuracy(percAccTrain, percAccTest)


def reportAccuracy(inputs, sz, ih_WMAT, ho_WMAT, gt, complete):
    """
    If reportAccuracy is passed a 0 for the 'complete' parameter, then the training
        is not yet complete and the function will solely compute and return the accuracy.
    If reportAccuracy is passed a 1 for the 'complete' parameter, then the training
        has completed and the function will return both the accuracy and confusion matrix.
    Returns the network accuracy given training and test data, network is not modified.
    """

    if complete == 1:
        confPred = np.zeros(sz)
        confAct = np.zeros(sz)

    accuracy = 0
    for i in range(sz):
        hidden = np.linalg.multi_dot([ih_WMAT, inputs[i]])
        hidden = activation(hidden)
        output = np.linalg.multi_dot([ho_WMAT, hidden])
        output = activation(output)

        index = np.argmax(output)
        predicted = np.zeros(10)
        predicted[index] = 1

        if complete == 1:
            confPred[i] = index
            confAct[i] = np.argmax(gt[i])

        if np.array_equal(gt[i], predicted) == True:
            accuracy += 1

    if complete == 1:
        pdPred = pd.Series(confPred, name='Predicted')
        pdAct = pd.Series(confAct, name='Actual')
        return accuracy, pd.crosstab(pdPred, pdAct)
    else:
        return accuracy


def epoch(inTrain, szTrain, ih_WMAT, ho_WMAT, gtTrain):
    """
    Runs a single epoch, iterating over all images in the training set.
    Returns the newly updated weights and the accuracy over the training set.
    """

    for i in range(szTrain):
        hidden = np.linalg.multi_dot([ih_WMAT, inTrain[i]])
        hidden = activation(hidden)
        output = np.linalg.multi_dot([ho_WMAT, hidden])
        output = activation(output)

        errorOutput, errorHidden = computeError(
            output, hidden, gtTrain[i], ho_WMAT.transpose())

        ih_WMAT, ho_WMAT = adjustWeights(
            hidden, ho_WMAT, errorOutput, inTrain[i], ih_WMAT, errorHidden)

    return ih_WMAT, ho_WMAT


def computeError(output, hidden, target, ho_tp_WMAT):
    """
    Computes error terms for both the hidden and output layer.
    The error term for the hidden layer is dependent upon the output layer error.
    Therefore outputer layer error term must always be computed first.
    Returns COLUMN vectors of hidden layer error terms and output layer error terms.
    """
    errorOutput = output*(1-output)*(target-output)

    backElement = np.zeros(NUMBER_OF_HIDDEN)
    for j in range(NUMBER_OF_HIDDEN):
        backElement[j] = np.dot(ho_tp_WMAT[j], errorOutput)

    errorHidden = hidden*(1-hidden)*(backElement)
    return np.array([errorOutput]).transpose(), np.array([errorHidden]).transpose()


def adjustWeights(hidden, ho_WMAT, errorOutput, inputs, ih_WMAT, errorHidden):
    """
    Adjusting input-to-hidden and hidden-to-output wieghts based on
        the computed error term for each layer (done in computeError() function).
    Returns the updated weights matrices.
    """
    ho_WMAT = ho_WMAT + (errorOutput*LEARNING_RATE*hidden)
    ih_WMAT = ih_WMAT + (errorHidden*LEARNING_RATE*inputs)

    return ih_WMAT, ho_WMAT


def activation(inputs):
    """
    Data that passes through a perceptron during the feed-forward stage will
        be fed into this activation function.
    The activation function used for this network is the sigmoid function.
    Numpy vector notation is used such that the sigmoid function is applied to
        all inputs in one line (without an apparent loop).
    Assume x is the input to this function,
    The returned value will be f(x) = 1 / (1 + e^(-x)), where e is eulers number.
    """

    inputs = 1/(1 + np.exp(-inputs))
    return inputs


def preprocess(inTrain, szTrain, inTest, szTest):
    """
    The preprocess function accomplpishes the following:
        1. Shuffles both the training and the test input data
        2. Extract the image labels into ground truth vectors (one hot encoded from labels)
        3. Normalize training and test image pixel values
        4. Overwrites label pixel value to be 1 (this is the bias value)
    Returns both ground truth vectors and the newly modified input data.
    """

    np.random.shuffle(inTrain)
    np.random.shuffle(inTest)

    gtTrain = np.zeros((szTrain, NUMBER_OF_CLASSES))
    gtTest = np.zeros((szTest, NUMBER_OF_CLASSES))
    gtTrain[np.arange(szTrain), inTrain.transpose()[0]] = 1
    gtTest[np.arange(szTest), inTest.transpose()[0]] = 1

    inTrain = inTrain/255
    inTest = inTest/255
    inTrain.transpose()[0] = 1
    inTest.transpose()[0] = 1

    return gtTrain, gtTest, inTrain, inTest


def load():
    """
    Extracts training and test images from file into numpy arrays.
    Initializes both weight matrices with values ranging from -0.05 to 0.05.
    Weight matrix dimensions are reliant on the number of classifications as
        well as the number of hidden units and input pixels from the images.
    These are constant parameters which can be changed at the beginning of
        the file.
    Returns the training and test data in vector form, as well as the weight
        matrices and the sizes of the training and test data vectors.
    """

    inTrain = np.array([[int(num) for num in lines.split(',')]
                        for lines in open(TRAINING_DATA).readlines()])
    inTest = np.array([[int(num) for num in lines.split(',')]
                       for lines in open(TESTING_DATA).readlines()])
    szTrain = len(inTrain)
    szTest = len(inTest)

    if EXAMPLES_PERCENT < 1:
        inTrain = inTrain[np.random.choice(
            inTrain.shape[0], int(EXAMPLES_PERCENT*szTrain), False)]
        szTrain = len(inTrain)

    NUMBER_OF_INPUTS = len(inTrain[0])

    ih_WMAT = np.random.uniform(-0.05, 0.05,
                                (NUMBER_OF_HIDDEN, NUMBER_OF_INPUTS))
    ho_WMAT = np.random.uniform(-0.05, 0.05,
                                (NUMBER_OF_CLASSES, NUMBER_OF_HIDDEN))

    return inTrain, inTest, ih_WMAT, ho_WMAT, szTrain, szTest


def plotAccuracy(percAccTrain, percAccTest):
    """
    Generates a plot based on the accuracy of the network.
    Two lines are drawn to represent accuracy of the training and test data respectively.
    Plot saved to machine as PNG.
    """

    xAxis = list(range(0, NUMBER_OF_EPOCHS))
    plt.figure()
    plt.suptitle("Handwritten Digit Classification")

    plt.ylim(0, MAX_ACCURACY)
    plt.ylabel("Accuracy (%)")

    plt.xlim(0, NUMBER_OF_EPOCHS)
    plt.xlabel("Epochs")

    plt.plot(xAxis, percAccTrain, label="Training")
    plt.plot(xAxis, percAccTest, label="Test")
    plt.legend()

    fileName = f'Plot_EP{EXAMPLES_PERCENT}_HU{NUMBER_OF_HIDDEN}_LR{LEARNING_RATE}'.replace(
        ".", "")
    fileName = f'{fileName}.png'
    print(f'Plot saved as: {fileName}')
    plt.savefig(fileName, format='png')


if __name__ == '__main__':
    main()
