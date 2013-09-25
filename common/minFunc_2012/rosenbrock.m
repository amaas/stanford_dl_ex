function fxy = rosenbrock(x,y)
fxy=((1-x).^2)+(100*((y-(x.^2)).^2));
return;
end