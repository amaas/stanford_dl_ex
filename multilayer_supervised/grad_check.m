function average_error = grad_check(fun, theta0, num_checks, varargin)

delta=1e-4;
sum_error=0;

fprintf(' Iter       i             err             rate');
fprintf('           g               g_est               f');
fprintf('           deltaf           f-delta               f+delta\n');

check_type = 'rand';
% check_type = 'all';
if strcmpi(check_type, 'all')
    num_checks = numel(theta0);
end
for i=1:num_checks
    T = theta0;
    if strcmpi(check_type, 'all')
        j = i;
    else
        j = randsample(numel(T),1);
    end
    T0=T; T0(j) = T0(j)-delta;
    T1=T; T1(j) = T1(j)+delta;
    
    [f,g] = fun(T, varargin{:});
    f0 = fun(T0, varargin{:});
    f1 = fun(T1, varargin{:});
    
    g_est = (f1-f0) / (2*delta);
    error = abs(g(j) - g_est);
    error_rate = error / (abs(g(j)) + 1e-11);
    if error > delta || error_rate > delta
        fprintf('% 5d  % 6d % 15g % 15g % 15f % 15f % 15f %15f %15f %15f\n', ...
            i, j, error, error_rate, g(j), g_est, f, f1-f0, f0, f1);
    end
    sum_error = sum_error + error;
end

average_error = sum_error/num_checks;
