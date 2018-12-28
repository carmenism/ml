function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X * theta);
n = length(theta);

cost_reg = (lambda / (2 * m)) * sum(theta(2:n) .^ 2);
% theta(2:n) because we do not include theta_0 in regularization

% COST
J = (1.0 / m) * (-y' * log(h_theta) - (1 - y') * log (1 - h_theta)) + cost_reg;

grad_reg = (lambda / m) * theta;
grad_reg(1) = 0;

% GRADIENT
grad = (1.0 / m) * X' * (h_theta - y) + grad_reg; 

% =============================================================

end
