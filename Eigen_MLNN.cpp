/*
 * Kobe Davis
 * Prof. Doliotis
 * CS 445
 * 13 March 2020
 *
 * Final Project
 */

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <utility>
#include <cmath>
#include <chrono>
#include "Dense"

using namespace Eigen;
using namespace std;
using namespace chrono;

/* Hyper-Parameters and Important Constants */
const auto NUMBER_OF_CLASSES = 10;
const auto NUMBER_OF_INPUTS = 785;
const auto NUMBER_OF_EPOCHS = 50;
const auto NUMBER_OF_HIDDEN = 40;
const auto EXAMPLES_PERCENT = 1;
const double LEARNING_RATE = 0.001;
constexpr auto TRAINING_DATA = "mnist_train.csv";
constexpr auto TESTING_DATA = "mnist_test.csv";

constexpr int rowsTrain = 60000;
constexpr int rowsTest =  10000;
constexpr int cols = 785;

/* Function Prototypes */
double reportAcc(const MatrixXd & data, const MatrixXd & gt, MatrixXd & i2h, MatrixXd & h2o);
double epoch(const MatrixXd & data, const MatrixXd & gt, MatrixXd & i2h, MatrixXd & h2o);
const MatrixXd oneHot(MatrixXd & data);
MatrixXd shuffle(MatrixXd & data);
template<typename M, const int rows, const int cols> constexpr M load(const char*);

int main() {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    /* Confirm with user that parameters are correct */
    //cout << "\n-------------------------------------------------" << endl;
    //cout << "Number of classifications: " << NUMBER_OF_CLASSES << endl;
    //cout << "Number of epochs: " << NUMBER_OF_EPOCHS << endl;
    //cout << "Number of hidden units: " << NUMBER_OF_HIDDEN << endl;
    //cout << "Proportion of training examples used: " << EXAMPLES_PERCENT << endl;
    //cout << "Learning rate: " << LEARNING_RATE << endl;
    //cout << "Training file: " << TRAINING_DATA << endl;
    //cout << "Test file: " << TESTING_DATA << endl;
    //cout << "-------------------------------------------------\n" << endl;

    //cout << "Please confirm that this information is correct by pressing <ENTER>." << endl;
    //cin.get();

    cout << "\nLoading data..." << endl;

    /* Generate random values betwen -0.05 and 0.05 for weight matrices */
    default_random_engine generator;
    uniform_real_distribution<double> distribution(-0.05, 0.05);
    auto uniform = [&]{return distribution(generator);};

    /* Generate weight matrices */
    MatrixXd i2h = MatrixXd::NullaryExpr(NUMBER_OF_HIDDEN, NUMBER_OF_INPUTS, uniform);
    MatrixXd h2o = MatrixXd::NullaryExpr(NUMBER_OF_CLASSES, NUMBER_OF_HIDDEN, uniform);

    /* Load input data from external files */
    MatrixXd inTrain = load<MatrixXd, rowsTrain, cols>(TRAINING_DATA);
    MatrixXd inTest = load<MatrixXd, rowsTest, cols>(TESTING_DATA);
    cout << "\tNumber of training examples: " << inTrain.rows() << endl;
    cout << "\tNumber of test examples: " << inTest.rows() << endl;

    cout << "\nPreprocessing..." << endl;

    /* Randomly shuffle input data */
    MatrixXd pinTrain = shuffle(inTrain);
    MatrixXd pinTest = shuffle(inTest);

    /*  The two following lines accomplish three tasks:        
     *      1. One-Hot encode ground truth labels for input data 
     *      2. Normalize input data between 0 and 1              
     *      3. Overwrite ground truth labels with bias value
     */
    const MatrixXd gtTrain = oneHot(pinTrain);
    const MatrixXd gtTest = oneHot(pinTest);

    cout << "Beginning epochs...\n" << endl;

    /*  This loop contains the bulk of this program's work                                       
     *  Each iteration performs one epoch and an accuracy measurement                            
     *  An epoch performs feedforward and backpropagation for all inputs                         
     *  Test and Training accuracies are returned from reportAcc and epoch respectively 
     */
    for(int i = 0; i < NUMBER_OF_EPOCHS; ++i) {
        cout << "Epoch " << i << endl;

        /* Test set prediction accuracy */
        int accTest = reportAcc(pinTest.transpose(), gtTest.transpose(), i2h, h2o);

        /* Network training and Training set prediction accuracy */
        int accTrain = epoch(pinTrain.transpose(), gtTrain.transpose(), i2h, h2o);

        cout << "\tTraining accuracy: " << round(100*accTrain)/100 << "%" << endl;
        cout << "\tTest accuracy: " << round(100*accTest)/100 << "%" << endl;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2-t1);

    cout << "Time taken: " << time_span.count() << " seconds" << endl;

    return 0;
}

/*  The function reportAcc determines the prediction accuracy of the network on the Test set
 *  The network is not modified here
 *  Function parameter explanation:
 *      1. data: Test set of input data
 *      2. gt:   One-Hot encoded ground truth labels
 *      3. i2h:  Weight matrix for input layer to hidden layer
 *      4. h2o:  Weight matrix for hidden layer to output layer
 *  Returns prediction accuracy as a percentage
 */
double reportAcc(const MatrixXd & data, const MatrixXd & gt, MatrixXd & i2h, MatrixXd & h2o) {
    double correct = 0;
    auto activation = [](double x){return 1/(1+exp(-x));};
    const double iterations = data.cols();
    for(int i = 0; i < iterations; ++i) {
        /* Feed-Forward and Activation */
        MatrixXd hidden = (i2h * data.col(i)).unaryExpr(activation);
        MatrixXd output = (h2o * hidden).unaryExpr(activation);

        /* Record Accuracy */
        int index = 0;
        output.col(0).maxCoeff(&index);
        if(gt.col(i)[index] == 1)
            ++correct;
    }
    return (correct/iterations) * 100;
}

/*  The function epoch trains the network by performing feedworward and backpropogation
 *  As well as determining the prediction accuracy of the network on the Training set
 *  The network is modified here, more specifically, the weight matrices are updated
 *  Function parameter explanation:
 *      1. data: Test set of input data
 *      2. gt:   One-Hot encoded ground truth labels
 *      3. i2h:  Weight matrix for input layer to hidden layer
 *      4. h2o:  Weight matrix for hidden layer to output layer
 *  Returns prediction accuracy as a percentage
 */
double epoch(const MatrixXd & data, const MatrixXd & gt, MatrixXd & i2h, MatrixXd & h2o) {
    double correct = 0;
    auto activation = [&](double x){return 1/(1+exp(-x));};
    const double iterations = data.cols();
    for(int i = 0; i < iterations; ++i) {
        /* Feed-Forward and Activation */
        MatrixXd hidden = (i2h * data.col(i)).unaryExpr(activation);
        MatrixXd output = (h2o * hidden).unaryExpr(activation);

        /* Record Accuracy */
        int index = 0;
        output.col(0).maxCoeff(&index);
        if(gt.col(i)[index] == 1)
            ++correct;

        /* Error Terms */
        const MatrixXd errOut = (output-output.cwiseProduct(output)).cwiseProduct(gt.col(i)-output);
        const MatrixXd errHid = (hidden-hidden.cwiseProduct(hidden)).cwiseProduct(h2o.transpose()*errOut);

        /* Adjust Weights */
        h2o += (errOut * LEARNING_RATE) * hidden.transpose();
        i2h += (errHid * LEARNING_RATE) * data.col(i).transpose();
    }
    return (correct/iterations) * 100;
}

/*  The function oneHot converts ground truth labels into a One-Hot encoded format
 *  Additionally, input data is normalized between 0 and 1
 *  And ground truth labels in the input data are overwritten by the bias value
 *  Function parameter explanation:
 *      1. data: Input data to the neural network
 *  Returns One-Hot encoded ground truth labels for the corresponding input data
 */
const MatrixXd oneHot(MatrixXd & data) {
    MatrixXd gt = MatrixXd::Zero(data.rows(), NUMBER_OF_CLASSES);
    for(int i = 0; i < data.rows(); ++i)
        gt(i,(int)data(i,0)) = 1;
    data.array() /= 255;
    data.col(0).array() = 1;
    return gt;
}

/*  The function shuffle receives input data and returns a shuffled copy of it
 *  Function paramater explanation:
 *      1. data: Input data to the neural network
 *  Returns a shuffled copy of the input data
 */
MatrixXd shuffle(MatrixXd & data) {
    vector<int> indices(data.rows());
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    return MatrixXd(data(indices, all));
}

/*  The function load extracts input data from an external file and loads it into an Eigen matrix
 *  Function parameter explanation:
 *      1. filepath: The filesystem path to the external file containing the input data
 *  Template parameter explanation:
 *      1. M:    The type of matrix used to store input data
 *      2. rows: The number of rows to initialize for the returned matrix
 *      3. cols: The number of columns to initialize for the returned matrix
 *  Returns an Eigen matrix containing the input data to the neural network
 *
 *  Further context:
 *      A function template became necessary to avoid computing matrix dimensions
 *      during runtime. Function parameters (in C++) can not be determined
 *      to be compile-time constants. Since determining the dimensions of the
 *      input matrices for the Test and Training sets during compile-time resulted
 *      in better performance, finding a way to do this in a function became desirable.
 *      It happens that template parameters for function templates CAN be determined
 *      to be compile-time constants when passed into a function. Hence a
 *      function template was used as a means to pass rows and columns into the
 *      function as compile-time constants, while still maintaining the generalizability
 *      of a function (as opposed to copy and pasting code to handle both the Test
 *      and Training sets differently).
 */
template<typename M, const int rows, const int cols>
constexpr M load(const char* filepath) { 
    string line;
    vector<double> values;
    ifstream file(filepath);

    while(getline(file, line)) {
        stringstream lineStream(line);
        string val;
        while(getline(lineStream, val, ','))
            values.push_back(stod(val));
    }
    return Map<const Matrix<typename M::Scalar, rows, cols, RowMajor>>(values.data(), rows, cols);
}
