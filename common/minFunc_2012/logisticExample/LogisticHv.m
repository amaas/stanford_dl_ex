function [Hv] = LogisticHv(v,w,X,y)
% v(feature,1) - vector that we will multiply Hessian by
% w(feature,1)
% X(instance,feature)
% y(instance,1)

sig = 1./(1+exp(-y.*(X*w)));
Hv = X.'*(sig.*(1-sig).*(X*v));
