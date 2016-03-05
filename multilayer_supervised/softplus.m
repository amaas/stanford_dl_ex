function y = softplus(x)

y = log(1 + exp(x));

high = find( x > 20 );
y(high) = x(high);