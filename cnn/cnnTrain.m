function cnnTrain(options)
%% Convolution Neural Network Exercise

%  Instructions
%  ------------
%
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

if nargin < 1
    learning_rate_schedule = 'half_per_epoch';
end

USE_GPU = 0;
dataset = 'mnist';
dataset = 'svhn';

% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

%% Load images and labels
switch dataset
    case 'mnist'
        % Load MNIST Train
        addpath ../common/;
        images = loadMNISTImages('../common/train-images-idx3-ubyte');
        images = reshape(images,imageDim,imageDim,[]);
        labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
        labels(labels==0) = 10; % Remap 0 to 10
        
        % mean normalization
        data_mean = mean(images, 3);
        data_std = std(images, 0, 3);
        data_std(data_std == 0) = 1;
        images = bsxfun(@minus, images, data_mean);
    case 'svhn'
        % Load SVHN Train
        load('../common/svhh_train_32_32_zeromean_mergeChannel');
        imageDim = 32;
end

% Sampling the data
numImages = size(images, 3);
% numImages = 60000;
% numImages = 10000;
images = images(:, :, 1:min(numImages, end));
labels = labels(1:min(numImages, end), :);

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

% % Transfer to GPU
if USE_GPU
    device = gpuDevice(1);
    device.reset();
    images = gpuArray(images);
    labels = gpuArray(labels);
    theta = gpuArray(theta);
end

%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG = false;  % set this to true to check gradient
% DEBUG = true;
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db_numFilters = 2;
    db_filterDim = 9;
    switch dataset
        case 'mnist'
            db_poolDim = 5;
        case 'svhn'
            db_poolDim = 6;
    end
    numDebugImages = 11; % better to be different from the numClasses
    db_images = images(:,:,1:numDebugImages);
    db_labels = labels(1:numDebugImages);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
        db_poolDim,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
        db_filterDim,db_numFilters,db_poolDim);
    
    
    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
        db_labels,numClasses,db_filterDim,...
        db_numFilters,db_poolDim), db_theta);
    
    % Use this to visually compare the gradients side by side
    num = numel(grad);
    for n = 1:num
        ratio = abs(grad(n) - numGrad(n)) / (abs(grad(n)) + 1e-6);
        if ratio > 1e-4
            fprintf('%d %10f %10f %10f\n', n, grad(n), numGrad(n), ratio);
        end
    end
    % Should be small. In our implementation, these values are usually
    % less than 1e-9.
    diff = norm(numGrad-grad)/norm(numGrad+grad)
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    return;
end

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

switch dataset
    case 'mnist'
        % Load mnist test
        testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
        testImages = reshape(testImages,imageDim,imageDim,[]);
        testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
        testLabels(testLabels==0) = 10; % Remap 0 to 10
        
        testImages = bsxfun(@minus, testImages, data_mean);
    case 'svhn'
        % % Load SVHN test
        test = load('../common/svhh_test_32_32_zeromean_mergeChannel');
        testImages = test.images;
        testLabels = test.labels;
end

% % Transfer to GPU
if USE_GPU
    testImages = gpuArray(testImages);
    testLabels = gpuArray(testLabels);
end

%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x, y, z, numClasses, filterDim, ...
    numFilters, poolDim), theta, images, labels, options, testImages, ...
    testLabels, numClasses, filterDim, numFilters, poolDim);

%%======================================================================
%% STEP 4: Test

% [~, cost, preds]=cnnCost(opttheta, testImages, testLabels, numClasses, ...
%     filterDim, numFilters, poolDim, true);
% 
% acc = 100 * sum(preds==testLabels) / length(preds);
% 
% % Accuracy should be around 97.4% after 3 epochs
% fprintf('Accuracy is %f\n',acc);