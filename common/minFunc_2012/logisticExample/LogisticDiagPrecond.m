function [m] = LogisticHv(v,w,X,y)
% v(feature,1) - vector that we will apply diagonal preconditioner to
% w(feature,1)
% X(instance,feature)
% y(instance,1)

sig = 1./(1+exp(-y.*(X*w)));

% Compute diagonals of Hessian
sig = sig.*(1-sig);
for i = 1:length(w)
   h(i,1) = (sig.*X(:,i))'*X(:,i);
end

% Apply preconditioner
m = v./h;

% Exact preconditioner
%H = X'*diag(sig.*(1-sig))*X;
%m = H\v;
