function [t,x_new,f_new,g_new,funEvals,H] = ArmijoBacktrack(...
    x,t,d,f,fr,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,funObj,varargin)
% [t,x_new,f_new,g_new,funEvals,H] = ArmijoBacktrack(...
%    x,t,d,f,fr,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,funObj,varargin)
%
% Backtracking linesearch to satisfy Armijo condition
%
% Inputs:
%   x: starting location
%   t: initial step size
%   d: descent direction
%   f: function value at starting location
%   fr: reference function value (usually funObj(x))
%   gtd: directional derivative at starting location
%   c1: sufficient decrease parameter
%   debug: display debugging information
%   LS_interp: type of interpolation
%   progTol: minimum allowable step length
%   doPlot: do a graphical display of interpolation
%   funObj: objective function
%   varargin: parameters of objective function
%
% Outputs:
%   t: step length
%   f_new: function value at x+t*d
%   g_new: gradient value at x+t*d
%   funEvals: number function evaluations performed by line search
%   H: Hessian at initial guess (only computed if requested)
%
% recet change: LS changed to LS_interp and LS_multi

% Evaluate the Objective and Gradient at the Initial Step
if nargout == 6
    [f_new,g_new,H] = funObj(x + t*d,varargin{:});
else
    [f_new,g_new] = funObj(x+t*d,varargin{:});
end
funEvals = 1;

while f_new > fr + c1*t*gtd || ~isLegal(f_new)
    temp = t;
    
    if LS_interp == 0 || ~isLegal(f_new)
        % Ignore value of new point
        if debug
            fprintf('Fixed BT\n');
        end
        t = 0.5*t;
    elseif LS_interp == 1 || ~isLegal(g_new)
        % Use function value at new point, but not its derivative
        if funEvals < 2 || LS_multi == 0 || ~isLegal(f_prev)
            % Backtracking w/ quadratic interpolation based on two points
            if debug
                fprintf('Quad BT\n');
            end
            t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot,0,t);
        else
            % Backtracking w/ cubic interpolation based on three points
            if debug
                fprintf('Cubic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot,0,t);
        end
    else
        % Use function value and derivative at new point
        
        if funEvals < 2 || LS_multi == 0 || ~isLegal(f_prev)
            % Backtracking w/ cubic interpolation w/ derivative
            if debug
                fprintf('Grad-Cubic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d],doPlot,0,t);
        elseif ~isLegal(g_prev)
            % Backtracking w/ quartic interpolation 3 points and derivative
            % of two
            if debug
                fprintf('Grad-Quartic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d; t_prev f_prev sqrt(-1)],doPlot,0,t);
        else
            % Backtracking w/ quintic interpolation of 3 points and derivative
            % of two
            if debug
                fprintf('Grad-Quintic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d; t_prev f_prev g_prev'*d],doPlot,0,t);
         end
    end
    
    % Adjust if change in t is too small/large
    if t < temp*1e-3
        if debug
            fprintf('Interpolated Value Too Small, Adjusting\n');
        end
        t = temp*1e-3;
    elseif t > temp*0.6
        if debug
            fprintf('Interpolated Value Too Large, Adjusting\n');
        end
        t = temp*0.6;
    end

    % Store old point if doing three-point interpolation
    if LS_multi
        f_prev = f_new;
        t_prev = temp;
        if LS_interp == 2
            g_prev = g_new;
        end
    end
    
    if ~saveHessianComp && nargout == 6
        [f_new,g_new,H] = funObj(x + t*d,varargin{:});
    else
        [f_new,g_new] = funObj(x + t*d,varargin{:});
    end
    funEvals = funEvals+1;

    % Check whether step size has become too small
    if max(abs(t*d)) <= progTol
        if debug
            fprintf('Backtracking Line Search Failed\n');
        end
        t = 0;
        f_new = f;
        g_new = g;
        break;
    end
end

% Evaluate Hessian at new point
if nargout == 6 && funEvals > 1 && saveHessianComp
    [f_new,g_new,H] = funObj(x + t*d,varargin{:});
    funEvals = funEvals+1;
end

x_new = x + t*d;

end
