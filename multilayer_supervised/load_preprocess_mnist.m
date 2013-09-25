function [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist()
%% TODO ensure this is consistent with common loaders
% assumes relative paths to the common directory
% assumes common directory on paty for access to load functions
% adds 1 to the labels to make them 1-indexed

data_train = loadMNISTImages('../common/train-images-idx3-ubyte');
labels_train = loadMNISTLabels(['../common/train-labels-idx1-ubyte']);
labels_train  = labels_train + 1;

data_test = loadMNISTImages('../common/t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels(['../common/t10k-labels-idx1-ubyte']);
labels_test = labels_test + 1;

