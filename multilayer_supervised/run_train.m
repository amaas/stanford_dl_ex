% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

data_mean = mean(data_train, 2);
data_std = std(data_train, 0, 2);
data_std(data_std == 0) = 1;
data_train = bsxfun(@minus, data_train, data_mean);
data_test = bsxfun(@minus, data_test, data_mean);

% data_train = bsxfun(@rdivide, data_train, data_std);
% data_test = bsxfun(@rdivide, data_test, data_std);

%% Benchmark accuracy and speed to decide to do which normalization
%%%%%%%hiddenLay = [100]
% 
% normalization = None
% test accuracy: 0.972700
% train accuracy: 0.983683
% 
% normalization = 0 mean
% test accuracy: 0.974200
% train accuracy: 0.984983
% 
% normalization = 0 mean, 1 std
% test accuracy: 0.965400
% train accuracy: 0.980117
% Elapsed time is 93.720963 seconds.
% 
% 
%%%%%%%hiddenLay = [100, 100]
% 
% normalization = None
% test accuracy: 0.974200
% train accuracy: 0.984750
% Elapsed time is 172.219214 seconds.
% 
% normalization = 0 mean
% test accuracy: 0.973700
% train accuracy: 0.985400
% Elapsed time is 111.577532 seconds.
% 
% normalization = 0 mean, 1 std
% test accuracy: 0.964400
% train accuracy: 0.980517
% Elapsed time is 102.581952 seconds.


%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [100, 100, ei.output_dim];
% ei.layer_sizes = [100, ei.output_dim];

% scaling parameter for l2 weight regularization penalty
ei.lambda = 1e-5;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% ei.activation_fun = 'logistic';
ei.activation_fun = 'relu';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
num_checks = 1000;
pred_only = false;
% grad_check(@supervised_dnn_cost, params, num_checks, ei, ...
%     data_train(:, 1:100), labels_train(1:100, :), pred_only);
numSamples = 30000;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params, options, ei, data_train(:, 1:numSamples), ...
    labels_train(1:numSamples, :));

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
