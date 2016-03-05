function [ output_args ] = mergeChannelsSvhn( input_args )
%MERGECHANNELSSVHN Summary of this function goes here
%   Detailed explanation goes here

suffix = 'pcawhiten';
suffix = 'zeromean';

load(['svhn_train_32x32_' suffix]);
images = mergeChannel(X);
labels = y;
save(['svhh_train_32_32_' suffix '_mergeChannel.mat'], 'images', 'labels');

load(['svhn_test_32x32_' suffix]);
images = mergeChannel(X);
labels = y;
save(['svhh_test_32_32_' suffix '_mergeChannel.mat'], 'images', 'labels');

end

function [images] = mergeChannel(X)
height = size(X, 1);
width = size(X, 2);
numImages = size(X, 4);
images = zeros(height, width, numImages);
parfor num = 1:numImages
    images(:, :, num) = mean(X(:, :, :, num), 3);
end

end