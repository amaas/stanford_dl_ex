function [l,d,perm] = mchol(A,mu)
% [l,d,perm] = mchol(A,mu)
% Compute the Gill-Murray modified LDL factorization of A,

if nargin < 2
    mu = 1e-12;
end

n = size(A,1);
l = eye(n);
d = zeros(n,1);
perm = 1:n;

for i = 1:n
    c(i,i) = A(i,i);
end

% Compute modification parameters
gamma = max(abs(diag(A)));
xi = max(max(abs(setdiag(A,0))));
delta = mu*max(gamma+xi,1);
if n > 1
    beta = sqrt(max([gamma xi/sqrt(n^2-1) mu]));
else
    beta = sqrt(max([gamma mu]));
end

for j = 1:n
    
    % Find q that results in Best Permutation with j
    [maxVal maxPos] = max(abs(diag(c(j:end,j:end))));
    q = maxPos+j-1;
    
    % Permute d,c,l,a
    d([j q]) = d([q j]);
    perm([j q]) = perm([q j]);
    c([j q],:) = c([q j],:);
    c(:,[j q]) = c(:,[q j]);
    l([j q],:) = l([q j],:);
    l(:,[j q]) = l(:,[q j]);
    A([j q],:) = A([q j],:);
    A(:,[j q]) = A(:,[q j]);
    
    for s = 1:j-1
        l(j,s) = c(j,s)/d(s);
    end
    for i = j+1:n
        c(i,j) = A(i,j) - sum(l(j,1:j-1).*c(i,1:j-1));
    end
    theta = 0;
    if j < n
        theta = max(abs(c(j+1:n,j)));
    end
    d(j) = max([abs(c(j,j)) (theta/beta)^2 delta]);
    if j < n
        for i = j+1:n
            c(i,i) = c(i,i) - (c(i,j)^2)/d(j);
        end
    end
end