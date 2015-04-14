clear all; 
hold off;

% Load training set.
load mnist38_train;

 %Uncomment these lines to use a small training set of 5 samples from 
 %each class. 
%Xtrain = Xtrain([1:5, 101:105],:); 
%Ttrain = Ttrain([1:5, 101:105]); 

% The digits can be visualized using the following command (the 10th
% training data).
%imagesc(reshape(Xtrain(10,:),[28 28])', [0 1]); colormap('gray');

N = size(Xtrain, 1); % Row Number of training sample, input number
X = [Xtrain'; ones(1,N)]; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
t = Ttrain; % Target binary vector. 
M = size(Xtrain, 2); % Column Number of training sample, input dimension

% Load validation set.
load mnist38_validate; 
Nv = size(Xvalidate, 1); 
Xv = [Xvalidate'; ones(1,Nv)]; 
tv = Tvalidate; 

% Load test set.
load mnist38_test;
Nt = size(Xtest, 1); % Row Number of training sample, input number
Xt = [Xtest'; ones(1,Nt)]; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
tt = Ttest; % Target binary vector. 

% TODO: Hyperparameter lambda which controls the prior on w.
lambda = 0;

% TODO: Learning rate. 
alpha = 0.0002;

% TODO: Perform gradient descent, run for MaxIter number of iterations.
MaxIter = 2000;

% Initialize the weight vector using samples from a normal distribution. 
w = randn(M+1, 1); %返回一个(M+1)*1的随机项矩阵 均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布

figure;   %打开窗口；
axes;     %在当前窗口创建一个包含默认属性的坐标系；
axis([0 2000]);  %创建x坐标0～100；y坐标0～1000；
set(gca,'Xtick',0:100:2000);   %在x轴上，标记0:20:100的数列；
hold on;
for n = 1:MaxIter
    [E, dE] = logistic_regression_objective(w, X, t, lambda);

    [Ev,dEv]=logistic_regression_objective(w,Xv,tv,lambda);
    
    plot (n, E,".g"); 
    hold on;
    plot(n, Ev,".b");
    hold on;
  
    w = w - alpha*dE;
    % TODO: You need to modify this loop and possibly add additional loops
    % to produce the required plots.
end

legend("Training Set","Validation Set");
xlabel("No. of Iteration");
ylabel("Negative Log Likelihood(E)");
title("ML Estimation lambda=0, alpha=0.0002, MaxIter=2000");
y = logistic_reg_prediction(w, X);
% Calculate the portion of correctly classified training data.
frac = sum((y > 0.5) - t == 0)/N;
 
yv = logistic_reg_prediction(w, Xv);
% Calculate the portion of correctly classified validation data.
frac_v = sum((yv > 0.5) - tv == 0)/Nv;

yt=logistic_reg_prediction(w,Xt);
frac_t = sum((yt > 0.5) - tt == 0)/Nt;
  
fprintf(1, 'Itr:%4d E:%4.2f Success Rate(Train):%2.2f;(Validate):%2.2f;(Test):%2.2f\n', n, E, frac, frac_v,frac_t);
