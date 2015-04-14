function [ E, dE ] = logistic_regression_objective( w, X, t, lambda )
% Calculate the objective and gradient for logistic regression.
%
%   Inputs:
%       w:  (M+1) x 1 vector of weights including the bias.
%       X:  (M+1) x N input data matrix, each column is a data point.
%       t:  N x 1 binary target vector.
%       lambda: hyperparameter for regularization, set to 0 for 
%               maximum likelihood.
%
%   Outputs:
%       E:  Scalar value for the objective function.
%       dE: (M+1) x 1 gradient of the objective (w.r.t. w).
% TODO: Implement this function.



    neg_Log_L=-sum(t'.*log(sigmoid(w'*X))+((ones(size(t'))-t').*log(ones(size(w'*X))-sigmoid(w'*X))));

    E=neg_Log_L+0.5*lambda*(w'*w);
                        
    dE=((sigmoid(w'*X)-t')*X').'+lambda.*w;

end

