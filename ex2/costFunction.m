function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

grad = theta;
% Note: grad should have the same dimensions as theta
%

%NOTE BY ANIMESH
%When you use fminunc() then dont need to write loop inside this function
%just write the derivative(gradient) term and cost function

%Gradient
hypothesis = sigmoid(X*grad);
val = hypothesis - y;
grad = (1/m).*(X'*val);


%Cost function

hypothesis = sigmoid(X*theta);
val1 = log(hypothesis);
val2 = log(1-hypothesis);
J = -((y')*val1 + (1-y')*val2)./m;


end
