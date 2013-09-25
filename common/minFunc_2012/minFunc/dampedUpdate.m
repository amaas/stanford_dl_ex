function [old_dirs,old_stps,Hdiag,Bcompact] = lbfgsUpdate(y,s,corrections,debug,old_dirs,old_stps,Hdiag)

%B0 = eye(length(y))/Hdiag;
S = old_dirs(:,2:end);
Y = old_stps(:,2:end);
k = size(Y,2);
L = zeros(k);
for j = 1:k
    for i = j+1:k
        L(i,j) = S(:,i)'*Y(:,j);
    end
end
D = diag(diag(S'*Y));
N = [S/Hdiag Y];
M = [S'*S/Hdiag L;L' -D];

ys = y'*s;
Bs = s/Hdiag - N*(M\(N'*s)); % Product B*s
sBs = s'*Bs;

eta = .02;
if ys < eta*sBs
    if debug
        fprintf('Damped Update\n');
    end
    theta = min(max(0,((1-eta)*sBs)/(sBs - ys)),1);
    y = theta*y + (1-theta)*Bs;
end


numCorrections = size(old_dirs,2);
if numCorrections < corrections
    % Full Update
    old_dirs(:,numCorrections+1) = s;
    old_stps(:,numCorrections+1) = y;
else
    % Limited-Memory Update
    old_dirs = [old_dirs(:,2:corrections) s];
    old_stps = [old_stps(:,2:corrections) y];
end

% Update scale of initial Hessian approximation
Hdiag = (y'*s)/(y'*y);