function features = feedfowardRICA(filterDim, poolDim, numFilters, images, W)
% feedfowardRICA Returns the convolution of the features given by W with
% the given images. It should be very similar to cnnConvolve.m+cnnPool.m 
% in the CNN exercise, except that there is no bias term b, and the pooling
% is RICA-style square-square-root pooling instead of average pooling.
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W    - W should be the weights learnt using RICA
%         W is of shape (filterDim,filterDim,numFilters)
%
% Returns:
%  features - matrix of convolved and pooled features in the form
%                      features(imageRow, imageCol, featureNum, imageNum)
global params;
numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

features = zeros(convDim / poolDim, ...
        convDim / poolDim, numFilters, numImages);
poolMat = ones(poolDim);
% Instructions:
%   Convolve every filter with every image just like what you did in
%   cnnConvolve.m to get a response.
%   Then perform square-square-root pooling on the response with 3 steps:
%      1. Square every element in the response
%      2. Sum everything in each pooling region
%      3. add params.epsilon to every element before taking element-wise square-root
%      (Hint: use poolMat similarly as in cnnPool.m)



for imageNum = 1:numImages
  if mod(imageNum,500)==0
    fprintf('forward-prop image %d\n', imageNum);
  end
  for filterNum = 1:numFilters

    filter = zeros(8,8); % You should replace this
    % Form W, obtain the feature (filterDim x filterDim) needed during the
    % convolution
    %%% YOUR CODE HERE %%%

    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter),2);
      
    % Obtain the image
    im = squeeze(images(:, :, imageNum));

    resp = zeros(convDim, convDim); % You should replace this
    % Convolve "filter" with "im" to find "resp"
    % be sure to do a 'valid' convolution
    %%% YOUR CODE HERE %%%
    % Then, apply square-square-root pooling on "resp" to get the hidden
    % activation "act"
    act = zeros(convDim / poolDim, convDim / poolDim); % You should replace this
    %%% YOUR CODE HERE %%%
    features(:, :, filterNum, imageNum) = act;
  end
end


end

