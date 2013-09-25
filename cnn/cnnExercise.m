%% Convolution and Pooling Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  convolution and pooling exercise. In this exercise, you will only
%  need to modify cnnConvolve.m and cnnPool.m. You will not need to modify
%  this file.

%%======================================================================
%% STEP 0: Initialization and Load Data
%  Here we initialize some parameters used for the exercise.

imageDim = 28;         % image dimension

filterDim = 8;          % filter dimension
numFilters = 100;         % number of feature maps

numImages = 60000;    % number of images

poolDim = 3;          % dimension of pooling region

% Here we load MNIST training images
addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,numImages);

W = randn(filterDim,filterDim,numFilters);
b = rand(numFilters);

%%======================================================================
%% STEP 1: Implement and test convolution
%  In this step, you will implement the convolution and test it on
%  on a small part of the data set to ensure that you have implemented
%  this step correctly.

%% STEP 1a: Implement convolution
%  Implement convolution in the function cnnConvolve in cnnConvolve.m

%% Use only the first 8 images for testing
convImages = images(:, :, 1:8); 

% NOTE: Implement cnnConvolve in cnnConvolve.m first!
convolvedFeatures = cnnConvolve(filterDim, numFilters, convImages, W, b);

%% STEP 1b: Checking your convolution
%  To ensure that you have convolved the features correctly, we have
%  provided some code to compare the results of your convolution with
%  activations from the sparse autoencoder

% For 1000 random points
for i = 1:1000   
    filterNum = randi([1, numFilters]);
    imageNum = randi([1, 8]);
    imageRow = randi([1, imageDim - filterDim + 1]);
    imageCol = randi([1, imageDim - filterDim + 1]);    
   
    patch = convImages(imageRow:imageRow + filterDim - 1, imageCol:imageCol + filterDim - 1, imageNum);

    feature = sum(sum(patch.*W(:,:,filterNum)))+b(filterNum);
    feature = 1./(1+exp(-feature));
    
    if abs(feature - convolvedFeatures(imageRow, imageCol,filterNum, imageNum)) > 1e-9
        fprintf('Convolved feature does not match test feature\n');
        fprintf('Filter Number    : %d\n', filterNum);
        fprintf('Image Number      : %d\n', imageNum);
        fprintf('Image Row         : %d\n', imageRow);
        fprintf('Image Column      : %d\n', imageCol);
        fprintf('Convolved feature : %0.5f\n', convolvedFeatures(imageRow, imageCol, filterNum, imageNum));
        fprintf('Test feature : %0.5f\n', feature);       
        error('Convolved feature does not match test feature');
    end 
end

disp('Congratulations! Your convolution code passed the test.');

%%======================================================================
%% STEP 2: Implement and test pooling
%  Implement pooling in the function cnnPool in cnnPool.m

%% STEP 2a: Implement pooling
% NOTE: Implement cnnPool in cnnPool.m first!
pooledFeatures = cnnPool(poolDim, convolvedFeatures);

%% STEP 2b: Checking your pooling
%  To ensure that you have implemented pooling, we will use your pooling
%  function to pool over a test matrix and check the results.

testMatrix = reshape(1:64, 8, 8);
expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
                  mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];
            
testMatrix = reshape(testMatrix, 8, 8, 1, 1);
        
pooledFeatures = squeeze(cnnPool(4, testMatrix));

if ~isequal(pooledFeatures, expectedMatrix)
    disp('Pooling incorrect');
    disp('Expected');
    disp(expectedMatrix);
    disp('Got');
    disp(pooledFeatures);
else
    disp('Congratulations! Your pooling code passed the test.');
end
