function [y] = precondUpper(r,U,D)
y = U \ (D .* (U' \ r));