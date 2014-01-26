function output = sigm(input)

output = 1./(1+exp(-input));

output(input > 13) = 1;
output(input < -13) = 0;