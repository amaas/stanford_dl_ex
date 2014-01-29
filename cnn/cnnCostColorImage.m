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

imageDim = size(images, 1); % height/width of image
numChannels = size(images, 3);
numImages = size(images, 4); % number of images

weightDecay = 1e-3; % regularization
USE_WEIGHT_DECAY = 1;

activationType = 'relu'; % Elapsed time is 395.22254 seconds. Accuracy is 98.34, 98.62(mean-normalized data)
% activationType = 'sigmoid'; % Elapsed time is 436.91066 seconds. Accuracy is 97.00


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
poolingIndex = 1 : poolDim : size(conv2(conv2(images(:, :, 1, 1), Wc_rotated(:, :, 1), 'valid'), meanPoolingFilter, 'valid'), 1);
parfor imageNum = 1 : numImages
    for filterNum = 1 : numFilters
        for channelNum = 1 : numChannels
            image = images(:, :, channelNum, imageNum);
            %         filter = Wc_rotated(:, :, filterNum);
            %         filteredImage = conv2(image, filter, 'valid');
            filteredImage = conv2(image, Wc_rotated(:, :, filterNum), 'valid') + bc(filterNum);
            
            switch activationType
                case 'relu'
                    filteredImage = max(filteredImage, 0); % relu
                case 'sigmoid'
                    filteredImage = sigmoid(filteredImage); % sigmoid
            end
            activations(:, :, filterNum, imageNum) = activations(:, :, filterNum, imageNum) + filteredImage;
            pooledImage = conv2(filteredImage, meanPoolingFilter, 'valid');
            activationsPooled(:, :, filterNum, imageNum) = activationsPooled(:, :, filterNum, imageNum) + pooledImage(poolingIndex, poolingIndex);
        end
    end
end
activations = activations / numChannels;
activationsPooled = activationsPooled / numChannels;

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

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%

% (numClasses x hiddenSize) * (hiddenSize x numImages)
activationsSoftmax = Wd * activationsPooledReshaped + repmat(bd, 1, numImages);
% activationsSoftmax = bsxfun(@plus, Wd * activationsPooledReshaped, bd);
activationsSoftmax = bsxfun(@minus, activationsSoftmax, max(activationsSoftmax));
activationsSoftmax = exp(activationsSoftmax);
probs = bsxfun(@rdivide, activationsSoftmax, sum(activationsSoftmax));

% activationsSoftmax = Wd * activationsPooledReshaped + repmat(bd , 1, numImages);
% activationsSoftmax = bsxfun(@minus, activationsSoftmax, max(activationsSoftmax, [], 1));
% activationsSoftmax = exp(activationsSoftmax);
% probs = bsxfun(@rdivide, activationsSoftmax, sum(activationsSoftmax)); %why rdivide?

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
labelIndex = sub2ind(size(activationsSoftmax), labels', 1:numImages);
% log_probs = log(probs);
% cost = -sum(log_probs(labelIndex));
if USE_GPU
    onehotLabels = gpuArray.zeros(size(activationsSoftmax));
else
    onehotLabels = zeros(size(activationsSoftmax));
end
onehotLabels(labelIndex) = 1;
cost = -sum(sum(onehotLabels .* log(probs)));

if USE_WEIGHT_DECAY
    weightDecayCost = .5 * weightDecay * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));
else
    weightDecayCost = 0;
end
cost = cost / numImages + weightDecayCost;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end

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
errorsSoftmax = probs - onehotLabels;
errorsSoftmax = errorsSoftmax / numImages;

% Backpropagate through the mean pooling layer
errorsPooled = Wd' * errorsSoftmax;
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);

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
% % if USE_GPU
% %     errorsConvolution = gpuArray.zeros(convDim,convDim,numFilters,numImages);
% % else
% %     errorsConvolution = zeros(convDim,convDim,numFilters,numImages);
% % end
% % % parfor imageNum = 1:numImages
% % for imageNum = 1:numImages
% %     for filterNum = 1:numFilters
% %         e = errorsPooling(:, :, filterNum, imageNum) ;
% %         switch activationType
% %             case 'relu'
% %                 a = activations(:, :, filterNum, imageNum) > 0; % relu
% %                 errorsConvolution(:, :, filterNum, imageNum) = e .* a ; % relu
% %             case 'sigmoid'
% %                 a = activations(:, :, filterNum, imageNum); % sigmoid
% %                 errorsConvolution(:, :, filterNum, imageNum) = e .* a .* (1 - a); % sigmoid
% %         end
% %     end
% % end

switch activationType
    case 'relu'
        errorsConvolution = errorsPooling .* (activations > 0); % relu derivative = x > 1
    case 'sigmoid'
        errorsConvolution = errorsPooling .* activations .* (1 - activations); % sigmoid derivative = x .* (1 - x)
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
Wd_grad = errorsSoftmax * activationsPooledReshaped';
if USE_WEIGHT_DECAY
    Wd_grad = Wd_grad + weightDecay * Wd;
end
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
    %     e = errorsConvolution(:, :, filterNum, :);
    e = errorsPooling(:, :, filterNum, :);
    bc_grad(filterNum) = sum(e(:));
end
parfor filterNum = 1 : numFilters
    % for filterNum = 1 : numFilters
    for imageNum = 1 : numImages
        e = errorsConvolution(:, :, filterNum, imageNum);
        %         e = errorsPooling(:, :, filterNum, imageNum);
        errorsConvolution(:, :, filterNum, imageNum) = rot90(e, 2);
        
        %         errorsPooling(:, :, filterNum, imageNum) = rot90(errorsPooling(:, :, filterNum, imageNum), 2);
    end
end
% if USE_GPU
%     Wc_gradFilter = gpuArray.zeros(size(Wc_grad, 1), size(Wc_grad, 2), numImages);
% else
%     Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2), numImages);
% end
for filterNum = 1 : numFilters
    Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
    %     parfor imageNum = 1 : numImages
    for imageNum = 1 : numImages
        for channelNum = 1 : numChannels
            image = images(:, :, channelNum, imageNum);
            %                 image = images(:, :, imageNum);
            %                 error = errorsPooling(:, :, filterNum, imageNum);
            %         %         Wc_grad(:, :, filterNum) = Wc_grad(:, :, filterNum) + conv2(image, error, 'valid');
            %         Wc_gradFilter(:, :, imageNum) = conv2(image, error, 'valid');
            %         Wc_gradFilter(:, :, imageNum) = conv2(images(:, :, imageNum), errorsPooling(:, :, filterNum, imageNum), 'valid');
            
            Wc_gradFilter = Wc_gradFilter + conv2(image, errorsConvolution(:, :, filterNum, imageNum), 'valid');
            %         Wc_gradFilter = Wc_gradFilter + conv2(image, error, 'valid');
        end
    end
    %     Wc_grad(:, :, filterNum) = sum(Wc_gradFilter, 3) / numImages + regularization;
    Wc_grad(:, :, filterNum) = Wc_gradFilter / numChannels;
end
if USE_WEIGHT_DECAY
    Wc_grad = Wc_grad + weightDecay * Wc;
end

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
