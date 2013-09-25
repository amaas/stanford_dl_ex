function [Hv] = autoHv(v,x,g,useComplex,funObj,varargin)
% [Hv] = autoHv(v,x,g,useComplex,funObj,varargin)
%
% Numerically compute Hessian-vector product H*v of funObj(x,varargin{:})
%  based on gradient values

if useComplex
    mu = 1e-150i;
else
    mu = 2*sqrt(1e-12)*(1+norm(x))/norm(v);
end
[f,finDif] = funObj(x + v*mu,varargin{:});
Hv = (finDif-g)/mu;