function [nll,g,H,T] = LogisticLoss(w,X,y)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);

Xw = X*w;
yXw = y.*Xw;

nll = sum(mylogsumexp([zeros(n,1) -yXw]));

if nargout > 1
    if nargout > 2
        sig = 1./(1+exp(-yXw));
        g = -X.'*(y.*(1-sig));
    else
        %g = -X.'*(y./(1+exp(yXw)));
        g = -(X.'*(y./(1+exp(yXw))));
    end
end

if nargout > 2
    H = X.'*diag(sparse(sig.*(1-sig)))*X;
end

if nargout > 3
    T = zeros(p,p,p);
    for j1 = 1:p
        for j2 = 1:p
            for j3 = 1:p
                T(j1,j2,j3) = sum(y(:).^3.*X(:,j1).*X(:,j2).*X(:,j3).*sig.*(1-sig).*(1-2*sig));
            end
        end
    end
end