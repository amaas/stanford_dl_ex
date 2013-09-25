
function c = bsxfunwrap(func, a, b)

global usegpu;

if usegpu
    if size(a,1) > 1 && size(b,1) == 1
        assert(size(a,2) == size(b,2), 'bsxfunwrap singleton dimensions dont agree');
        c = func(a, repmat(b, size(a,1), 1));
    elseif size(a,2) > 1 && size(b,2) == 1
        assert(size(a,1) == size(b,1), 'bsxfunwrap singleton dimensions dont agree');
        c = func(a, repmat(b, 1, size(a,2)));
    elseif size(b,1) > 1 && size(a,1) == 1
        assert(size(b,2) == size(a,2), 'bsxfunwrap singleton dimensions dont agree');
        c = func(repmat(a, size(b, 1), 1), b);
    elseif size(b,2) > 1 && size(a,2) == 1
        assert(size(b,1) == size(a,1), 'bsxfunwrap singleton dimensions dont agree');
        c = func(repmat(a, 1, size(b, 2)), b);
    else
        assert(size(a,1) == size(b,1), 'no bsxfun to do, bsxfunwrap dimensions dont agree');
        assert(size(a,2) == size(b,2), 'no bsxfun to do, bsxfunwrap dimensions dont agree');
        c = func(a, b);
    end
else
    c = bsxfun(func, a, b);
end

end