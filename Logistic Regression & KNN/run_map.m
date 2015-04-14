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

load mnist38_test; 
Nt = size(Xtest, 1); 
Xt = [Xtest'; ones(1,Nt)]; 
tt = Ttest; 

% TODO: Hyperparameter lambda which controls the prior on w.
lambda = [0,1,3,5,100];
%lambda = [5]; % lambda=5 is found to be effective.

% TODO: Learning rate. 
alpha = 0.0002;

% TODO: Perform gradient descent, run for MaxIter number of iterations.
MaxIter = 2000;

% Initialize the weight vector using samples from a normal distribution. 
%w = randn(M+1, 1); %返回一个(M+1)*1的随机项矩阵 均值为0，方差 σ^2 = 1，标准差σ = 1的正态分布

for j=1:5
%for j=1:1
    sum_E=0; sum_Ev=0;
    sum_frac=0; sum_frac_v=0; sum_frac_t=0;
    for i=1:10
      w = randn(M+1, 1);
      for n = 1:MaxIter
          [E, dE] = logistic_regression_objective(w, X, t, lambda(j));
          w = w - alpha*dE;
          [Ev, dEv] = logistic_regression_objective(w, Xv, tv, lambda(j));
      end
      y = logistic_reg_prediction(w, X);
      frac = sum((y > 0.5) - t == 0)/N;
      yv = logistic_reg_prediction(w, Xv);
      frac_v = sum((yv > 0.5) - tv == 0)/Nv;
      
      yt = logistic_reg_prediction(w, Xt);
      frac_t = sum((yt > 0.5) - tt == 0)/Nt;
      
      
      E2=logistic_regression_objective(w, X, t, 0);
      
      Ev2=logistic_regression_objective(w, Xv, tv, 0);
      
      sum_E=sum_E+E2;
      sum_Ev=sum_Ev+Ev2;
      sum_frac=sum_frac+frac;
      sum_frac_v=sum_frac_v+frac_v;
      %sum_frac_t=sum_frac_t+frac_t;
    end
    plot (lambda(j), sum_frac/10,".g"); 
    %plot(lambda(j),sum_E/10,".g");
    hold on;
    plot (lambda(j), sum_frac_v/10,".b"); 
    %plot(lambda(j),sum_Ev/10,".b");
    hold on;
    fprintf('%d %2.2f %2.2f\n', j, sum_E/10, sum_Ev/10);
    %fprintf('final accuracy on test set %2.2f\n', sum_frac_t/10);
    
end
legend("Training Set","Validation Set");
xlabel("Lambda(λ)");
%ylabel("Negative Log Likelihood");
ylabel("Accuracy");
title("MAP Estimation w.r.t Lambda(λ) (alpha=0.0002, MaxIter=2000)");
hold on;