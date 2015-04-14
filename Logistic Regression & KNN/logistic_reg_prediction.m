function [ y ] = logistic_reg_prediction( w, X )
% Calculate the output of the logistic regression classifier.
%   
%   Inputs:
%       w:  (M+1) x 1 vector of weights including the bias.
%       X:  (M+1) x N input data matrix, each column is a data point.
%
%   Outputs:
%       y: N x 1 vector of probabilities (output of the classifier).
% TODO: Implement this function. 

y=sigmoid(w'*X).';


end

