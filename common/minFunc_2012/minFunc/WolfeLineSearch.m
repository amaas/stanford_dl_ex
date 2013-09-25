function [t,f_new,g_new,funEvals,H] = WolfeLineSearch(...
    x,t,d,f,g,gtd,c1,c2,LS_interp,LS_multi,maxLS,progTol,debug,doPlot,saveHessianComp,funObj,varargin)
%
% Bracketing Line Search to Satisfy Wolfe Conditions
%
% Inputs:
%   x: starting location
%   t: initial step size
%   d: descent direction
%   f: function value at starting location
%   g: gradient at starting location
%   gtd: directional derivative at starting location
%   c1: sufficient decrease parameter
%   c2: curvature parameter
%   debug: display debugging information
%   LS_interp: type of interpolation
%   maxLS: maximum number of iterations
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
%   H: Hessian at initial guess (only computed if requested

% Evaluate the Objective and Gradient at the Initial Step
if nargout == 5
    [f_new,g_new,H] = funObj(x + t*d,varargin{:});
else
    [f_new,g_new] = funObj(x+t*d,varargin{:});
end
funEvals = 1;
gtd_new = g_new'*d;

% Bracket an Interval containing a point satisfying the
% Wolfe criteria

LSiter = 0;
t_prev = 0;
f_prev = f;
g_prev = g;
gtd_prev = gtd;
nrmD = max(abs(d));
done = 0;

while LSiter < maxLS

    %% Bracketing Phase
    if ~isLegal(f_new) || ~isLegal(g_new)
        if debug
            fprintf('Extrapolated into illegal region, switching to Armijo line-search\n');
        end
        t = (t + t_prev)/2;
        % Do Armijo
        if nargout == 5
            [t,x_new,f_new,g_new,armijoFunEvals,H] = ArmijoBacktrack(...
                x,t,d,f,f,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,...
                funObj,varargin{:});
        else
            [t,x_new,f_new,g_new,armijoFunEvals] = ArmijoBacktrack(...
                x,t,d,f,f,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,...
                funObj,varargin{:});
        end
        funEvals = funEvals + armijoFunEvals;
        return;
    end


    if f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
        bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];
        break;
    elseif abs(gtd_new) <= -c2*gtd
        bracket = t;
        bracketFval = f_new;
        bracketGval = g_new;
        done = 1;
        break;
    elseif gtd_new >= 0
        bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];
        break;
    end
    temp = t_prev;
    t_prev = t;
    minStep = t + 0.01*(t-temp);
    maxStep = t*10;
    if LS_interp <= 1
        if debug
            fprintf('Extending Braket\n');
        end
        t = maxStep;
    elseif LS_interp == 2
        if debug
            fprintf('Cubic Extrapolation\n');
        end
        t = polyinterp([temp f_prev gtd_prev; t f_new gtd_new],doPlot,minStep,maxStep);
    elseif LS_interp == 3
        t = mixedExtrap(temp,f_prev,gtd_prev,t,f_new,gtd_new,minStep,maxStep,debug,doPlot);
    end
    
    f_prev = f_new;
    g_prev = g_new;
    gtd_prev = gtd_new;
    if ~saveHessianComp && nargout == 5
        [f_new,g_new,H] = funObj(x + t*d,varargin{:});
    else
        [f_new,g_new] = funObj(x + t*d,varargin{:});
    end
    funEvals = funEvals + 1;
    gtd_new = g_new'*d;
    LSiter = LSiter+1;
end

if LSiter == maxLS
    bracket = [0 t];
    bracketFval = [f f_new];
    bracketGval = [g g_new];
end

%% Zoom Phase

% We now either have a point satisfying the criteria, or a bracket
% surrounding a point satisfying the criteria
% Refine the bracket until we find a point satisfying the criteria
insufProgress = 0;
Tpos = 2;
LOposRemoved = 0;
while ~done && LSiter < maxLS

    % Find High and Low Points in bracket
    [f_LO LOpos] = min(bracketFval);
    HIpos = -LOpos + 3;

    % Compute new trial value
    if LS_interp <= 1 || ~isLegal(bracketFval) || ~isLegal(bracketGval)
        if debug
            fprintf('Bisecting\n');
        end
        t = mean(bracket);
    elseif LS_interp == 2
        if debug
            fprintf('Grad-Cubic Interpolation\n');
        end
        t = polyinterp([bracket(1) bracketFval(1) bracketGval(:,1)'*d
            bracket(2) bracketFval(2) bracketGval(:,2)'*d],doPlot);
    else
        % Mixed Case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nonTpos = -Tpos+3;
        if LOposRemoved == 0
            oldLOval = bracket(nonTpos);
            oldLOFval = bracketFval(nonTpos);
            oldLOGval = bracketGval(:,nonTpos);
        end
        t = mixedInterp(bracket,bracketFval,bracketGval,d,Tpos,oldLOval,oldLOFval,oldLOGval,debug,doPlot);
    end


    % Test that we are making sufficient progress
    if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
        if debug
            fprintf('Interpolation close to boundary');
        end
        if insufProgress || t>=max(bracket) || t <= min(bracket)
            if debug
                fprintf(', Evaluating at 0.1 away from boundary\n');
            end
            if abs(t-max(bracket)) < abs(t-min(bracket))
                t = max(bracket)-0.1*(max(bracket)-min(bracket));
            else
                t = min(bracket)+0.1*(max(bracket)-min(bracket));
            end
            insufProgress = 0;
        else
            if debug
                fprintf('\n');
            end
            insufProgress = 1;
        end
    else
        insufProgress = 0;
    end

    % Evaluate new point
    if ~saveHessianComp && nargout == 5
        [f_new,g_new,H] = funObj(x + t*d,varargin{:});
    else
        [f_new,g_new] = funObj(x + t*d,varargin{:});
    end
    funEvals = funEvals + 1;
    gtd_new = g_new'*d;
    LSiter = LSiter+1;

	armijo = f_new < f + c1*t*gtd;
    if ~armijo || f_new >= f_LO
        % Armijo condition not satisfied or not lower than lowest
        % point
        bracket(HIpos) = t;
        bracketFval(HIpos) = f_new;
        bracketGval(:,HIpos) = g_new;
        Tpos = HIpos;
    else
        if abs(gtd_new) <= - c2*gtd
            % Wolfe conditions satisfied
            done = 1;
        elseif gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
            % Old HI becomes new LO
            bracket(HIpos) = bracket(LOpos);
            bracketFval(HIpos) = bracketFval(LOpos);
            bracketGval(:,HIpos) = bracketGval(:,LOpos);
            if LS_interp == 3
                if debug
                    fprintf('LO Pos is being removed!\n');
                end
                LOposRemoved = 1;
                oldLOval = bracket(LOpos);
                oldLOFval = bracketFval(LOpos);
                oldLOGval = bracketGval(:,LOpos);
            end
        end
        % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;
        Tpos = LOpos;
	end

    if ~done && abs(bracket(1)-bracket(2))*nrmD < progTol
        if debug
            fprintf('Line-search bracket has been reduced below progTol\n');
        end
        break;
    end

end

%%
if LSiter == maxLS
    if debug
        fprintf('Line Search Exceeded Maximum Line Search Iterations\n');
    end
end

[f_LO LOpos] = min(bracketFval);
t = bracket(LOpos);
f_new = bracketFval(LOpos);
g_new = bracketGval(:,LOpos);



% Evaluate Hessian at new point
if nargout == 5 && funEvals > 1 && saveHessianComp
    [f_new,g_new,H] = funObj(x + t*d,varargin{:});
    funEvals = funEvals + 1;
end

end


%%
function [t] = mixedExtrap(x0,f0,g0,x1,f1,g1,minStep,maxStep,debug,doPlot);
alpha_c = polyinterp([x0 f0 g0; x1 f1 g1],doPlot,minStep,maxStep);
alpha_s = polyinterp([x0 f0 g0; x1 sqrt(-1) g1],doPlot,minStep,maxStep);
if alpha_c > minStep && abs(alpha_c - x1) < abs(alpha_s - x1)
    if debug
        fprintf('Cubic Extrapolation\n');
    end
    t = alpha_c;
else
    if debug
        fprintf('Secant Extrapolation\n');
    end
    t = alpha_s;
end
end

%%
function [t] = mixedInterp(bracket,bracketFval,bracketGval,d,Tpos,oldLOval,oldLOFval,oldLOGval,debug,doPlot);

% Mixed Case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nonTpos = -Tpos+3;

gtdT = bracketGval(:,Tpos)'*d;
gtdNonT = bracketGval(:,nonTpos)'*d;
oldLOgtd = oldLOGval'*d;
if bracketFval(Tpos) > oldLOFval
    alpha_c = polyinterp([oldLOval oldLOFval oldLOgtd
        bracket(Tpos) bracketFval(Tpos) gtdT],doPlot);
    alpha_q = polyinterp([oldLOval oldLOFval oldLOgtd
        bracket(Tpos) bracketFval(Tpos) sqrt(-1)],doPlot);
    if abs(alpha_c - oldLOval) < abs(alpha_q - oldLOval)
        if debug
            fprintf('Cubic Interpolation\n');
        end
        t = alpha_c;
    else
        if debug
            fprintf('Mixed Quad/Cubic Interpolation\n');
        end
        t = (alpha_q + alpha_c)/2;
    end
elseif gtdT'*oldLOgtd < 0
    alpha_c = polyinterp([oldLOval oldLOFval oldLOgtd
        bracket(Tpos) bracketFval(Tpos) gtdT],doPlot);
    alpha_s = polyinterp([oldLOval oldLOFval oldLOgtd
        bracket(Tpos) sqrt(-1) gtdT],doPlot);
    if abs(alpha_c - bracket(Tpos)) >= abs(alpha_s - bracket(Tpos))
        if debug
            fprintf('Cubic Interpolation\n');
        end
        t = alpha_c;
    else
        if debug
            fprintf('Quad Interpolation\n');
        end
        t = alpha_s;
    end
elseif abs(gtdT) <= abs(oldLOgtd)
    alpha_c = polyinterp([oldLOval oldLOFval oldLOgtd
        bracket(Tpos) bracketFval(Tpos) gtdT],...
        doPlot,min(bracket),max(bracket));
    alpha_s = polyinterp([oldLOval sqrt(-1) oldLOgtd
        bracket(Tpos) bracketFval(Tpos) gtdT],...
        doPlot,min(bracket),max(bracket));
    if alpha_c > min(bracket) && alpha_c < max(bracket)
        if abs(alpha_c - bracket(Tpos)) < abs(alpha_s - bracket(Tpos))
            if debug
                fprintf('Bounded Cubic Extrapolation\n');
            end
            t = alpha_c;
        else
            if debug
                fprintf('Bounded Secant Extrapolation\n');
            end
            t = alpha_s;
        end
    else
        if debug
            fprintf('Bounded Secant Extrapolation\n');
        end
        t = alpha_s;
    end

    if bracket(Tpos) > oldLOval
        t = min(bracket(Tpos) + 0.66*(bracket(nonTpos) - bracket(Tpos)),t);
    else
        t = max(bracket(Tpos) + 0.66*(bracket(nonTpos) - bracket(Tpos)),t);
    end
else
    t = polyinterp([bracket(nonTpos) bracketFval(nonTpos) gtdNonT
        bracket(Tpos) bracketFval(Tpos) gtdT],doPlot);
end
end