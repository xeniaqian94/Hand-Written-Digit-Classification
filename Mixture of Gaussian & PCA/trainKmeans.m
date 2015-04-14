clear all;
hold off;

load mnist38_train;

x=Xtrain;
K=2;
minVary = 0.01; 
N = size(x,1); % number of training data
T = size(x,2); % number of dimension

% Initialize the parameters
randConst = 1;
p = randConst+rand(K,1); p = p/sum(p);
vr = std(x,[],2).^2;
mu = kmeans(x, K, 5);