function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
    filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImages
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

USE_GPU = 0;

if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

weightDecay = 1e-3; % L2 regularization

activationType = 'relu';
% activationType = 'sigmoid';

% costFunction = 'ce'; % cross entropy
costFunction = 'nll'; % negative log likelyhood

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
    poolDim,numClasses);

% % Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
% Wc_grad = zeros(size(Wc));
% Wd_grad = zeros(size(Wd));
% bc_grad = zeros(size(bc));
% bd_grad = zeros(size(bd));


%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
% convDim * convDim
% numFilters
% numImages
% convDim * convDim * numFilters * numImages
if USE_GPU
    activations = gpuArray.zeros(convDim,convDim,numFilters,numImages);
else
    activations = zeros(convDim,convDim,numFilters,numImages);
end

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
if USE_GPU
    activationsPooled = gpuArray.zeros(outputDim,outputDim,numFilters,numImages);
else
    activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
end

%%% YOUR CODE HERE %%%

if USE_GPU
    meanPoolingFilter = gpuArray.ones(poolDim, poolDim);
    Wc_rotated = gpuArray.zeros(size(Wc));
else
    meanPoolingFilter = ones(poolDim, poolDim);
    Wc_rotated = zeros(size(Wc));
end
for filterNum = 1 : numFilters
    Wc_rotated(:, :, filterNum) = rot90(Wc(:, :, filterNum), 2);
end
areaOfPoolingFilter = poolDim ^ 2;
meanPoolingFilter = meanPoolingFilter / areaOfPoolingFilter;
poolingIndex = 1 : poolDim : size(conv2(conv2(images(:, :, 1), Wc_rotated(:, :, 1), 'valid'), meanPoolingFilter, 'valid'), 1);
parfor imageNum = 1 : numImages
    image = images(:, :, imageNum);
    for filterNum = 1 : numFilters
        %         filter = Wc_rotated(:, :, filterNum);
        %         filteredImage = conv2(image, filter, 'valid');
        
        filteredImage = conv2(image, Wc_rotated(:, :, filterNum), 'valid');
        
        switch activationType
            case 'relu'
                activations(:, :, filterNum, imageNum) = max(filteredImage, 0); % relu
            case 'sigmoid'
                activations(:, :, filterNum, imageNum) = sigmoid(filteredImage); % sigmoid
        end
        pooledImage = conv2(filteredImage, meanPoolingFilter, 'valid');
        activationsPooled(:, :, filterNum, imageNum) = pooledImage(poolingIndex, poolingIndex) + bc(filterNum);
    end
end

% activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
% activationsPooled = cnnPool(poolDim, activations);


% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooledReshaped = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% % numClasses x numImages for storing probability that each image belongs to
% % each class.
% probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%

% (numClasses x hiddenSize) * (hiddenSize x numImages)
activationsSoftmax = bsxfun(@plus, Wd * activationsPooledReshaped, bd);

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%

% numClasses * numImages
labelIndex = sub2ind(size(activationsSoftmax), labels', 1:numImages);
if USE_GPU
    onehotLabels = gpuArray.zeros(size(activationsSoftmax));
else
    onehotLabels = zeros(size(activationsSoftmax));
end
onehotLabels(labelIndex) = 1;
featureDim = 1;
switch costFunction
    case 'nll'
        probs = exp(bsxfun(@minus, activationsSoftmax, max(activationsSoftmax, [], featureDim)));
        probs = bsxfun(@rdivide, activationsSoftmax, sum(probs, featureDim));
        log_probs = log(probs);
        cost = -sum(log_probs(labelIndex));
    case 'ce'
        cost = -sum(sum(activationsSoftmax .* onehotLabels - log(1 + exp(activationsSoftmax)), 1), 2);
end
weightDecayCost = .5 * weightDecay * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));
cost = cost / numImages + weightDecayCost;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.
%  Use the kron function and a matrix of ones to do this upsampling
%  quickly.

%%% YOUR CODE HERE %%%
% Backpropagate through the softmax layer
switch costFunction
    case 'nll'
        errorsSoftmax = probs - onehotLabels;
    case 'ce'
        errorsSoftmax = sigmoid(activationsSoftmax) - onehotLabels;
end
errorsSoftmax = errorsSoftmax / numImages;

% errorsSoftmax = probs - targets; % cross entropy
% gradient = -(labels - sigm(output))/nSamples;

% Backpropagate through the mean pooling layer
errorsPooled = Wd' * errorsSoftmax;
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);
% size(errorsPooled)

if USE_GPU
    errorsPooling = gpuArray.zeros(convDim, convDim, numFilters, numImages);
    unpoolingFilter = gpuArray.ones(poolDim);
else
    errorsPooling = zeros(convDim, convDim, numFilters, numImages);
    unpoolingFilter = ones(poolDim);
end

poolArea = poolDim ^ 2;
unpoolingFilter = unpoolingFilter / poolArea;
parfor imageNum = 1:numImages
    % for imageNum = 1:numImages
    for filterNum = 1:numFilters
        e = errorsPooled(:, :, filterNum, imageNum);
        errorsPooling(:, :, filterNum, imageNum) = kron(e, unpoolingFilter);
        
        %         errorsPooling(:, :, filterNum, imageNum) = kron(errorsPooled(:, :, filterNum, imageNum), unpoolingFilter);
    end
end

% Backpropagate through the convolutional layer
if USE_GPU
    errorsConvolutional = gpuArray.zeros(convDim,convDim,numFilters,numImages);
else
    errorsConvolutional = zeros(convDim,convDim,numFilters,numImages);
end
% parfor imageNum = 1:numImages
for imageNum = 1:numImages
    for filterNum = 1:numFilters
        e = errorsPooling(:, :, filterNum, imageNum) ;
        switch activationType
            case 'relu'
                a = activations(:, :, filterNum, imageNum) > 0; % relu
                errorsConvolutional(:, :, filterNum, imageNum) = e .* a ; % relu
            case 'sigmoid'
                a = activations(:, :, filterNum, imageNum); % sigmoid
                errorsConvolutional(:, :, filterNum, imageNum) = e .* a .* (1 - a); % sigmoid
        end
    end
end


%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
% Gradient of the softmax layer
Wd_grad = errorsSoftmax * activationsPooledReshaped' + weightDecay * Wd;
bd_grad = sum(errorsSoftmax, 2);

% Gradient of the convolutional layer
if USE_GPU
    bc_grad = gpuArray.zeros(size(bc));
    Wc_grad = gpuArray.zeros(size(Wc));
else
    bc_grad = zeros(size(bc));
    Wc_grad = zeros(size(Wc));
end

% parfor filterNum = 1 : numFilters
for filterNum = 1 : numFilters
    e = errorsPooling(:, :, filterNum, :);
    bc_grad(filterNum) = sum(e(:));
end
parfor filterNum = 1 : numFilters
    % for filterNum = 1 : numFilters
    for imageNum = 1 : numImages
        e = errorsPooling(:, :, filterNum, imageNum);
        errorsPooling(:, :, filterNum, imageNum) = rot90(e, 2);
        
        %         errorsPooling(:, :, filterNum, imageNum) = rot90(errorsPooling(:, :, filterNum, imageNum), 2);
    end
end
% if USE_GPU
%     Wc_gradFilter = gpuArray.zeros(size(Wc_grad, 1), size(Wc_grad, 2), numImages);
% else
%     Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2), numImages);
% end
for filterNum = 1 : numFilters
    % for filterNum = 1 : numFilters
    %     parfor imageNum = 1 : numImages
    Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
    for imageNum = 1 : numImages
        %                 image = images(:, :, imageNum);
        %                 error = errorsPooling(:, :, filterNum, imageNum);
        %         %         Wc_grad(:, :, filterNum) = Wc_grad(:, :, filterNum) + conv2(image, error, 'valid');
        %         Wc_gradFilter(:, :, imageNum) = conv2(image, error, 'valid');
        %         Wc_gradFilter(:, :, imageNum) = conv2(images(:, :, imageNum), errorsPooling(:, :, filterNum, imageNum), 'valid');
        
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), errorsPooling(:, :, filterNum, imageNum), 'valid');
        %         Wc_gradFilter = Wc_gradFilter + conv2(image, error, 'valid');
    end
    %     Wc_grad(:, :, filterNum) = sum(Wc_gradFilter, 3) / numImages + regularization;
    Wc_grad(:, :, filterNum) = Wc_gradFilter;
end
Wc_grad = Wc_grad + weightDecay * Wc;

%% Unroll gradient into grad vector for minFunc
% sizeWc = size(Wc)
% sizeWd = size(Wd)
% sizebc = size(bc)
% sizebd = size(bd)
%
% sizeWc_grad = size(Wc_grad)
% sizeWd_grad = size(Wd_grad)
% sizebc_grad = size(bc_grad)
% sizebd_grad = size(bd_grad)
%
% size(grad)

% sizeWc_gradCol = size(Wc_grad(:))
% sizeWd_gradCol = size(Wd_grad(:))
% sizebc_gradCol = size(bc_grad(:))
% sizebd_gradCol = size(bd_grad(:))

grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
