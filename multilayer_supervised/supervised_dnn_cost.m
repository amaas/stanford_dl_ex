function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
    po = pred_only;
end;

%% reshape into network
numHidden = numel(ei.layer_sizes) - 1;
numSamples = size(data, 2);
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
stack = params2stack(theta, ei);
%% forward prop
%%% YOUR CODE HERE %%%
% size_data = size(data)
for l = 1 : numHidden
    if l > 1
        hAct{l} = stack{l}.W * hAct{l - 1} + repmat(stack{l}.b, 1, ...
            numSamples);
    else
        %                 l = l
        %                 size_W = size(stack{l}.W)
        %                 size_data = size(data)
        %                 size_b = size(stack{l}.b)
        hAct{l} = stack{l}.W * data + repmat(stack{l}.b, 1, numSamples);
    end
    switch ei.activation_fun
        case 'logistic'
            hAct{l} = sigmoid(hAct{l});
        case 'relu'
            hAct{l} = relu(hAct{l});
        case 'softmax'
            hAct{l} = softmax(hAct{l}, 1);
        case 'softplus'
            hAct{l} = softplus(hAct{l});
        case 'tanh'
            hAct{l} = tanh(hAct{l});
    end
    %     size_hAct_l = size(hAct{l})
end

l = numHidden+1;
y_hat = stack{l}.W * hAct{l - 1} + repmat(stack{l}.b, 1, numSamples);
y_hat = exp(y_hat);
hAct{l} = bsxfun(@rdivide, y_hat, sum(y_hat, 1));
[pred_prob pred_labels] = max(hAct{l});
pred_prob = hAct{l};
% if numel(labels) > 0
%     idx = pred_labels' == labels;
%     num_right_labels = size(idx)
% end

%% return here if only predictions desired.
if po
    cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
    grad = [];
    return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y_hat = log(hAct{numHidden+1});
index = sub2ind(size(y_hat), labels', 1:numSamples);
ceCost = -sum(y_hat(index));


% y_hat = exp(theta' * X); % (K - 1) * m
% y_hat = [y_hat; ones(1, size(y_hat, 2))]; % K * m

% y_hat_sum = sum(y_hat, 2); % K * 1
% y_hat_sum(end, :) = 1; % K * 1
% p_y = bsxfun(@rdivide, y_hat, y_hat_sum); % K * m
% A = log(p_y); % K * m
% index = sub2ind(size(y_hat), y, 1 : size(y_hat, 2));
% A(end, :) = 0;
% A(index); % m * 1
% f = -sum(A(index));
% indicator = zeros(size(p_y)); % K * m
% indicator(index) = 1;
% g = -X * (indicator - p_y)'; % K * n

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% Cross entroy gradient
targets = zeros(size(hAct{numHidden+1})); % numLabels * numSamples
targets(index) = 1;
gradInput = hAct{numHidden+1} - targets;

for l = numHidden+1 : -1 : 1
    if l > numHidden
        gradFunc = ones(size(gradInput));
    else
        switch ei.activation_fun
            case 'logistic'
                gradFunc = hAct{l} .* (1 - hAct{l});
            case 'relu'
                gradFunc = hAct{l} > 0;
                %         case 'softmax'
                %             hAct{l} = softmax(input, 1);
                %         case 'softplus'
                %             hAct{l} = softplus(input);
            case 'tanh'
                gradFunc = 1 - hAct{l} .^ 2;
        end
    end
    gradOutput = gradInput .* gradFunc;
    if l > 1
        gradStack{l}.W = gradOutput * hAct{l-1}';
    else
        gradStack{l}.W = gradOutput * data';
    end
    gradStack{l}.b = sum(gradOutput, 2);
    gradInput = stack{l}.W' * gradOutput;
end


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

wCost = 0;
for l = 1:numHidden+1
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2);
end

cost = ceCost + wCost;

% Computing the gradient of the weight decay.
for l = numHidden : -1 : 1
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



