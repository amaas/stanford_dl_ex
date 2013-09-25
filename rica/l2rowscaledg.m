function [grad] = l2rowscaledg(x,y,outderv, alpha)

normeps = 1e-5;
if (~exist('outderv','var')||isempty(outderv))
    error('Requires outderv of previous layer to compute gradient!');
end

epssumsq = sum(x.^2,2) + normeps;	

l2rows = sqrt(epssumsq)*alpha;

if (~exist('y','var')||isempty(y))
     y = bsxfunwrap(@rdivide,x,l2rows);
end

grad = bsxfunwrap(@rdivide, outderv, l2rows) - ...
       bsxfunwrap(@times, y, sum(outderv.*x, 2) ./ epssumsq);

