function [minPos,fmin] = polyinterp(points,doPlot,xminBound,xmaxBound)
% function [minPos] = polyinterp(points,doPlot,xminBound,xmaxBound)
%
%   Minimum of interpolating polynomial based on function and derivative
%   values
%
%   It can also be used for extrapolation if {xmin,xmax} are outside
%   the domain of the points.
%
%   Input:
%       points(pointNum,[x f g])
%       doPlot: set to 1 to plot, default: 0
%       xmin: min value that brackets minimum (default: min of points)
%       xmax: max value that brackets maximum (default: max of points)
%
%   set f or g to sqrt(-1) if they are not known
%   the order of the polynomial is the number of known f and g values minus 1

if nargin < 2
    doPlot = 0;
end

nPoints = size(points,1);
order = sum(sum((imag(points(:,2:3))==0)))-1;

xmin = min(points(:,1));
xmax = max(points(:,1));

% Compute Bounds of Interpolation Area
if nargin < 3
    xminBound = xmin;
end
if nargin < 4
    xmaxBound = xmax;
end

% Code for most common case:
%   - cubic interpolation of 2 points
%       w/ function and derivative values for both

if nPoints == 2 && order ==3 && doPlot == 0
    % Solution in this case (where x2 is the farthest point):
    %    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    %    d2 = sqrt(d1^2 - g1*g2);
    %    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    %    t_new = min(max(minPos,x1),x2);
    [minVal minPos] = min(points(:,1));
    notMinPos = -minPos+3;
    d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
    d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
    if isreal(d2)
        t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
        minPos = min(max(t,xminBound),xmaxBound);
    else
        minPos = (xmaxBound+xminBound)/2;
    end
    return;
end

% Constraints Based on available Function Values
A = zeros(0,order+1);
b = zeros(0,1);
for i = 1:nPoints
    if imag(points(i,2))==0
        constraint = zeros(1,order+1);
        for j = order:-1:0
            constraint(order-j+1) = points(i,1)^j;
        end
        A = [A;constraint];
        b = [b;points(i,2)];
    end
end

% Constraints based on available Derivatives
for i = 1:nPoints
    if isreal(points(i,3))
        constraint = zeros(1,order+1);
        for j = 1:order
            constraint(j) = (order-j+1)*points(i,1)^(order-j);
        end
        A = [A;constraint];
        b = [b;points(i,3)];
    end
end

% Find interpolating polynomial
[params,ignore] = linsolve(A,b);

% Compute Critical Points
dParams = zeros(order,1);
for i = 1:length(params)-1
    dParams(i) = params(i)*(order-i+1);
end

if any(isinf(dParams))
    cp = [xminBound;xmaxBound;points(:,1)].';
else
    cp = [xminBound;xmaxBound;points(:,1);roots(dParams)].';
end

% Test Critical Points
fmin = inf;
minPos = (xminBound+xmaxBound)/2; % Default to Bisection if no critical points valid
for xCP = cp
    if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
        fCP = polyval(params,xCP);
        if imag(fCP)==0 && fCP < fmin
            minPos = real(xCP);
            fmin = real(fCP);
        end
    end
end

% Plot Situation
if doPlot
    clf; hold on;

    % Plot Points
    plot(points(:,1),points(:,2),'b*');

    % Plot Derivatives
    for i = 1:nPoints
        if isreal(points(i,3))
            m = points(i,3);
            b = points(i,2) - m*points(i,1);
            plot([points(i,1)-.05 points(i,1)+.05],...
                [(points(i,1)-.05)*m+b (points(i,1)+.05)*m+b],'c.-');
        end
    end

    % Plot Function
    x = min(xmin,xminBound)-.1:(max(xmax,xmaxBound)+.1-min(xmin,xminBound)+.1)/100:max(xmax,xmaxBound)+.1;
    for i = 1:length(x)
        f(i) = polyval(params,x(i));
    end
    plot(x,f,'y');
    axis([x(1)-.1 x(end)+.1 min(f)-.1 max(f)+.1]);

    % Plot Minimum
    plot(minPos,fmin,'g+');
    if doPlot == 1
        pause(1);
    end
end