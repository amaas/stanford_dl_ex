function patches = samplePatches(rawImages, patchSize, numPatches)
% rawImages is of size (image width)*(image height) by number_of_images
% We assume that image width = image height
imWidth = sqrt(size(rawImages,1));
imHeight = imWidth;
numImages = size(rawImages,2);
rawImages = reshape(rawImages,imWidth,imHeight,numImages); 

% Initialize patches with zeros.  
patches = zeros(patchSize*patchSize, numPatches);

% Maximum possible start coordinate
maxWidth = imWidth - patchSize + 1;
maxHeight = imHeight - patchSize + 1;

% Sample!
for num = 1:numPatches
    x = randi(maxHeight);
    y = randi(maxWidth);
    img = randi(numImages);
    p = rawImages(x:x+patchSize-1,y:y+patchSize-1, img);
    patches(:,num) = p(:);
end
    

