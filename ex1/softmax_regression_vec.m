function [f,g] = softmax_regression_vec(theta, X, y)
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

y_hat = exp(theta' * X); % (K - 1) * m
y_hat = [y_hat; ones(1, size(y_hat, 2))]; % K * m

y_hat_sum = sum(y_hat, 2); % K * 1
y_hat_sum(end, :) = 1; % K * 1
p_y = bsxfun(@rdivide, y_hat, y_hat_sum); % K * m
A = log(p_y); % K * m
index = sub2ind(size(y_hat), y, 1 : size(y_hat, 2));
A(end, :) = 0;
A(index); % m * 1
f = -sum(A(index));
indicator = zeros(size(p_y)); % K * m
indicator(index) = 1;
g = -X * (indicator - p_y)'; % K * n

  g=g(:, 1:end - 1); 
  g = g(:); % make gradient a vector for minFunc
