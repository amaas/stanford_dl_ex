function [ xPCAWhitened ] = pcaWhiten( x )
%PCAWHITEN Summary of this function goes here
%   Detailed explanation goes here

[U, S, ~] = svd(x * x' / size(x, 2));

epsilon = 1e-1; 
%%% YOUR CODE HERE %%%
xPCAWhitened = diag(1./sqrt(diag(S) + epsilon)) * U' * x;

end

