function [opttheta] = minFuncSGD(funObj,theta,data,labels,...
    options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9

USE_GPU = 0;

%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
    'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
numSamples = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
if USE_GPU
    velocity = gpuArray.zeros(size(theta));
else
    velocity = zeros(size(theta));
end

%%======================================================================
%% SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(numSamples);
    ticEpoch = tic;
    for s=1 : minibatch : (numSamples - minibatch + 1)
        tic;
        it = it + 1;
        
        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;
        
        % get next randomly selected minibatch
        mb_data = data(:, :, rp(s : min(s + minibatch - 1, end)));
        mb_labels = labels(rp(s : min(s + minibatch - 1, end)));
        
        % evaluate the objective function on the next minibatch
        size_mb_data = size(mb_data);
        [cost grad] = funObj(theta, mb_data, mb_labels);
        
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        
        %%% YOUR CODE HERE %%%
        velocity = mom * velocity + alpha * grad;
        theta = theta - velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n', e, it, cost);
        toc;
%         if it > 1
%             break;
%         end
    end;
    fprintf('Epoch %d\n', e);
    toc(ticEpoch);
    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;
%     break;
end;

opttheta = theta;

end
