function [f,g,H] = taylorModel(d,f,g,H,T)

p = length(d);

fd3 = 0;
gd2 = zeros(p,1);
Hd = zeros(p);
for t1 = 1:p
    for t2 = 1:p
        for t3 = 1:p
            fd3 = fd3 + T(t1,t2,t3)*d(t1)*d(t2)*d(t3);

            if nargout > 1
                gd2(t3) = gd2(t3) + T(t1,t2,t3)*d(t1)*d(t2);
            end

            if nargout > 2
                Hd(t2,t3) = Hd(t2,t3) + T(t1,t2,t3)*d(t1);
            end
        end

    end
end

f = f + g'*d + (1/2)*d'*H*d + (1/6)*fd3;

if nargout > 1
    g = g + H*d + (1/2)*gd2;
end

if nargout > 2
    H = H + Hd;
end

if any(abs(d) > 1e5)
    % We want the optimizer to stop if the solution is unbounded
    g = zeros(p,1);
end