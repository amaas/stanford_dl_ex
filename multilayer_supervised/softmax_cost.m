function [cost, gradient] = softmax_cost(theta, X, y)
%
% Arguments:
%   theta - A vector containing the parameter values to optimize.
%       In minFunc, theta is reshaped to a long vector.  So we need to
%       resize it to an n-by-(num_classes-1) matrix.
%       Recall that we assume theta(:,num_classes) = 0.
%
%   X - The examples stored in a matrix.
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example.  y(j) is the j'th example's label.
%
m=size(X,2);
n=size(X,1);

% theta is a vector;  need to reshape to n x num_classes.
theta=reshape(theta, n, []);
num_classes=size(theta,2)+1;

% initialize objective value and gradient.
f = 0;
g = zeros(size(theta));

%
% TODO:  Compute the softmax objective function and gradient using vectorized code.
%        Store the objective function value in 'f', and the gradient in 'g'.
%        Before returning g, make sure you form it back into a vector with g=g(:);
%
%%% YOUR CODE HERE %%%

y_hat = (exp(theta' * X))'; % m * num_classes

y_hat_sum = sum(y_hat, 1); % 1 * num_classes
p_y = bsxfun(@rdivide, y_hat, y_hat_sum); % num_classes * m
A = log(p_y); % num_classes * m
% size(y_hat)
% size(1:size(y_hat, 1))
% size(y')
index = sub2ind(size(y_hat), 1 : size(y_hat, 1), y');
% size(A(index)); % m * 1
cost = -sum(A(index));


indicator = zeros(size(p_y)); % m * num_classes
indicator(index) = 1;
g = -X * (indicator - p_y); % num_classes * n

gradient = g(:, 1:end - 1);

% 
% function softmax(A, dim)
% s = ones(1, ndims(A));
% s(dim) = size(A, dim);
% 
% % First get the maximum of A.
% maxA = max(A, [], dim);
% expA = exp(A-repmat(maxA, s));
% softmaxA = expA ./ repmat(sum(expA,dim), s);


% softmaxOutput = softmax(output, 2);
% 	errors = sum(-log(softmaxOutput).*labels,2) + wdCost;
% 	gradient = (- labels + softmaxOutput)/nSamples;
