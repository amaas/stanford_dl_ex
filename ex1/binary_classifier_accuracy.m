function accuracy=binary_classifier_accuracy(theta, X,y)
  correct=sum(y == (sigmoid(theta'*X) > 0.5));
  accuracy = correct / length(y);
