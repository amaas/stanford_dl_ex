function [ output ] = sigmoid( input )

output = 1./(1+exp(-input));
