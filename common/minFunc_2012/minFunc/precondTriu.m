function [y] = precondUpper(r,U)
y = U \ (U' \ r);