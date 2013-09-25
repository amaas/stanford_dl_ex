clear all

nInst = 250;
nVars = 10;
X = randn(nInst,nVars);
w = randn(nVars,1);
y = sign(X*w + randn(nInst,1));

wTest = randn(nVars,1);

fprintf('Testing gradient using forward-differencing...\n');
order = 1;
derivativeCheck(@LogisticLoss,wTest,order,1,X,y);

fprintf('Testing gradient using central-differencing...\n');
derivativeCheck(@LogisticLoss,wTest,order,2,X,y);

fprintf('Testing gradient using complex-step derivative...\n');
derivativeCheck(@LogisticLoss,wTest,order,3,X,y);

fprintf('\n\n\n');
pause

fprintf('Testing Hessian using forward-differencing\n');
order = 2;
derivativeCheck(@LogisticLoss,wTest,order,1,X,y);

fprintf('Testing Hessian using central-differencing\n');
order = 2;
derivativeCheck(@LogisticLoss,wTest,order,2,X,y);

fprintf('Testing Hessian using complex-step derivative\n');
order = 2;
derivativeCheck(@LogisticLoss,wTest,order,3,X,y);

fprintf('\n\n\n');
pause

fprintf('Testing gradient using fastDerivativeCheck...\n');
order = 1;
fastDerivativeCheck(@LogisticLoss,wTest,order,1,X,y);
fastDerivativeCheck(@LogisticLoss,wTest,order,2,X,y);
fastDerivativeCheck(@LogisticLoss,wTest,order,3,X,y);

fprintf('\n\n\n');
pause

fprintf('Testing Hessian using fastDerivativeCheck...\n');
order = 2;
fastDerivativeCheck(@LogisticLoss,wTest,order,1,X,y);
fastDerivativeCheck(@LogisticLoss,wTest,order,2,X,y);
fastDerivativeCheck(@LogisticLoss,wTest,order,3,X,y);
