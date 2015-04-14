clear all; 
hold off;

% Load test set.
load mnist38_test;

N = size(Xtest, 1); % Row Number of training sample, input number
X = [Xtest'; ones(1,N)]; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
t = Ttest; % Target binary vector. 
M = size(Xtest, 2); % Column Number of training sample, input dimension

% TODO: Hyperparameter lambda which controls the prior on w.
lambda = 5;

% TODO: Learning rate. 
alpha = 0.0002;

% TODO: Perform gradient descent, run for MaxIter number of iterations.
MaxIter = 2000;

% Initialize the weight vector using samples from a normal distribution. 
%w = randn(M+1, 1); %返回一个(M+1)*1的随机项矩阵 均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布

w = randn(M+1, 1); %返回一个(M+1)*1的随机项矩阵 均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布

for n = 1:MaxIter
    [E, dE] = logistic_regression_objective(w, X, t, lambda);
    
    w = w - alpha*dE;
      
    % TODO: You need to modify this loop and possibly add additional loops
    % to produce the required plots.
end
y = logistic_reg_prediction(w, X);
% Calculate the portion of correctly classified training data.
frac = sum((y > 0.5) - t == 0)/N;
fprintf(1, 'Itr:%4d E:%4.2f Success Rate (Test):%2.2f\n', n, E, frac);