function [  ] = normalizeSvhn( USE_PCA_WHITEN )
%NORMALIZASVHN Summary of this function goes here
%   Detailed explanation goes here

if nargin < 1
    USE_PCA_WHITEN = 0;
end

if USE_PCA_WHITEN
    addpath('../pca');
    pacWhitenEpsilon = 1e-1;
end
load('svhn_train_32x32');
X = double(X) / 255;

numChannels = size(X, 3);
meanData = zeros(size(X, 1), size(X, 2), numChannels);
for nc = 1:numChannels
    meanData(:, :, nc) = mean(X(:, :, nc, :), 4);
end

X = bsxfun(@minus, X, meanData);
if USE_PCA_WHITEN
    eigenVectors = cell(numChannels);
    eigenValues = cell(numChannels);
    randsel = randi(size(X, 4), 100, 1); % A random selection of samples for visualization
    for nc = 1:numChannels
        xSingleChannel = reshape(squeeze(X(:, :, nc, :)), [], size(X, 4));
        [U, S, ~] = svd(xSingleChannel * xSingleChannel' / size(xSingleChannel, 2));
        xSingleChannel = diag(1./sqrt(diag(S) + pacWhitenEpsilon)) * U' * xSingleChannel;
        figure('name','Visualisation of PCA whitened training images');
        display_network(xSingleChannel(:, randsel) * 255);
        figure('name','Visualisation of ZCA whitened training images');
        display_network(U * xSingleChannel(:, randsel) * 255);
        X(:, :, nc, :) = reshape(xSingleChannel, size(X, 1), size(X, 2), 1, size(X, 4));
        eigenVectors{nc} = U;
        eigenValues{nc} = S;
    end
    save('svhn_train_32x32_pcawhiten.mat', 'X', 'y', 'meanData');
else
    save('svhn_train_32x32_zeromean.mat', 'X', 'y', 'meanData');
end

load('svhn_test_32x32');
X = double(X) / 255;
X = bsxfun(@minus, X, meanData);
if USE_PCA_WHITEN
    randsel = randi(size(X, 4), 100, 1); % A random selection of samples for visualization
    for nc = 1:numChannels
        xSingleChannel = reshape(squeeze(X(:, :, nc, :)), [], size(X, 4));
        U = eigenVectors{nc};
        S = eigenValues{nc};
        xSingleChannel = diag(1./sqrt(diag(S) + pacWhitenEpsilon)) * U' * xSingleChannel;
        figure('name','Visualisation of PCA whitened test images');
        display_network(xSingleChannel(:, randsel) * 255);
        figure('name','Visualisation of ZCA whitened test images');
        display_network(U * xSingleChannel(:, randsel) * 255);
        X(:, :, nc, :) = reshape(xSingleChannel, size(X, 1), size(X, 2), 1, size(X, 4));
    end
    save('svhn_test_32x32_pcawhiten.mat', 'X', 'y', 'meanData');
else
    save('svhn_test_32x32_zeromean.mat', 'X', 'y', 'meanData');
end

% delete(findall(0,'Type','figure'));

end

