function [x,k,res,negCurv] = cg(A,b,optTol,maxIter,verbose,precFunc,precArgs,matrixVectFunc,matrixVectArgs)
% [x,k,res,negCurv] =
% cg(A,b,optTol,maxIter,verbose,precFunc,precArgs,matrixVectFunc,matrixVect
% Args)
% Linear Conjugate Gradient, where optionally we use
% - preconditioner on vector v with precFunc(v,precArgs{:})
% - matrix multipled by vector with matrixVectFunc(v,matrixVectArgs{:})

if nargin <= 4
    verbose = 0;
end

x = zeros(size(b));
r = -b;

% Apply preconditioner (if supplied)
if nargin >= 7 && ~isempty(precFunc)
    y = precFunc(r,precArgs{:});
else
    y = r;
end

ry = r'*y;
p = -y;
k = 0;

res = norm(r);
done = 0;
negCurv = [];
while res > optTol & k < maxIter & ~done
    % Compute Matrix-vector product
    if nargin >= 9
        Ap = matrixVectFunc(p,matrixVectArgs{:});
    else
        Ap = A*p;
    end
    pAp = p'*Ap;

    % Check for negative Curvature
    if pAp <= 1e-16
        if verbose
            fprintf('Negative Curvature Detected!\n');
        end
        
        if nargout == 4
           if pAp < 0
              negCurv = p;
              return
           end
        end
        
        if k == 0
            if verbose
                fprintf('First-Iter, Proceeding...\n');
            end
            done = 1;
        else
            if verbose
                fprintf('Stopping\n');
            end
            break;
        end
    end

    % Conjugate Gradient
    alpha = ry/(pAp);
    x = x + alpha*p;
    r = r + alpha*Ap;
    
    % If supplied, apply preconditioner
    if nargin >= 7 && ~isempty(precFunc)
        y = precFunc(r,precArgs{:});
    else
        y = r;
    end
    
    ry_new = r'*y;
    beta = ry_new/ry;
    p = -y + beta*p;
    k = k + 1;

    % Update variables
    ry = ry_new;
    res = norm(r);
end
end
