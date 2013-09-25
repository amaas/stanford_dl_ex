function [params] = stack2params(stack)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.
% This is a non-standard version of the code to support conv nets
% it allows higher layers to have window sizes >= 1 of the previous layer
% If using a gpu pass inParams as your gpu datatype
% Setup the compressed param vector
params = [];

    
for d = 1:numel(stack)
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    params = [params ; stack{d}.W(:) ; stack{d}.b(:) ];
    
    % Check that stack is of the correct form
    assert(size(stack{d}.W, 1) == size(stack{d}.b, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.W, 1)) 'x1']);
     % no layer size constrain with conv nets
     if d < numel(stack)
        assert(mod(size(stack{d+1}.W, 2), size(stack{d}.W, 1)) == 0, ...
            ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
             ' should have matching sizes.']);
     end
end

end