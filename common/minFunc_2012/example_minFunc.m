% Runs various limited-memory solvers on 2D rosenbrock function for 25
% function evaluations
maxFunEvals = 25;

fprintf('Result after %d evaluations of limited-memory solvers on 2D rosenbrock:\n',maxFunEvals);

fprintf('---------------------------------------\n');
fprintf('x1 = %.4f, x2 = %.4f (starting point)\n',0,0);
fprintf('x1 = %.4f, x2 = %.4f (optimal solution)\n',1,1);
fprintf('---------------------------------------\n');

if exist('minimize') == 2
    % Minimize.m - conjugate gradient method
    x = minimize([0 0]', 'rosenbrock', -maxFunEvals);
    fprintf('x1 = %.4f, x2 = %.4f (minimize.m by C. Rasmussen)\n',x(1),x(2));
end

options = [];
options.display = 'none';
options.maxFunEvals = maxFunEvals;

% Steepest Descent
options.Method = 'sd';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with steepest descent)\n',x(1),x(2));

% Cyclic Steepest Descent
options.Method = 'csd';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with cyclic steepest descent)\n',x(1),x(2));

% Barzilai & Borwein
options.Method = 'bb';
options.bbType = 1;
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with spectral gradient descent)\n',x(1),x(2));

% Hessian-Free Newton
options.Method = 'newton0';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with Hessian-free Newton)\n',x(1),x(2));

% Hessian-Free Newton w/ L-BFGS preconditioner
options.Method = 'pnewton0';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with preconditioned Hessian-free Newton)\n',x(1),x(2));

% Conjugate Gradient
options.Method = 'cg';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with conjugate gradient)\n',x(1),x(2));

% Scaled conjugate Gradient
options.Method = 'scg';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with scaled conjugate gradient)\n',x(1),x(2));

% Preconditioned Conjugate Gradient
options.Method = 'pcg';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with preconditioned conjugate gradient)\n',x(1),x(2));

% Default: L-BFGS (default)
options.Method = 'lbfgs';
x = minFunc(@rosenbrock,[0 0]',options);
fprintf('x1 = %.4f, x2 = %.4f (minFunc with limited-memory BFGS - default)\n',x(1),x(2));

fprintf('---------------------------------------\n');







