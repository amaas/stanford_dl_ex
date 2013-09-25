function lse = mylogsumexp(b)
% does logsumexp across columns
B = max(b,[],2);
lse = log(sum(exp(b-repmat(B,[1 size(b,2)])),2))+B;

% Old version that used repmatC
%lse = log(sum(exp(b-repmatC(B,[1 size(b,2)])),2))+B;
end