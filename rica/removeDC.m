% Removes DC component from image patches
% Data given as a matrix where each patch is one column vectors
% That is, the patches are vectorized.

function [Y,meanX]=removeDC(X, dim);

% Subtract local mean gray-scale value from each patch in X to give output Y
if nargin == 1
    dim = 1;
end

meanX = mean(X,dim);

if dim==1
    Y = X-meanX(ones(size(X,1),1),:);
else
    Y = X-meanX(:,ones(size(X,2),1));
end

return;
