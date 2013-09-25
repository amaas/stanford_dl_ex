function accuracy=binary_classifier_accuracy(theta, X,y)
  [~,labels] = max(theta'*X, [], 1);

  correct=sum(y == labels);
  accuracy = correct / length(y);
