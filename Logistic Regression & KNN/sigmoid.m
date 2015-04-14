function [ y ] = sigmoid( a )
% Calculate the sigmoid function of the input.
%
%   Inputs: 
%       a: A vector of real values.
%
%   Outputs: 
%       y: sigmoid(a), a vector of the same size as a.

y = 1.0 ./ (ones(size(a)) + exp(-a)); 

end

