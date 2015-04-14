clear all; 
hold off;

load mnist38_train;
N = size(Xtrain, 1); % Row Number of training sample, input number
X = [Xtrain']; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
t = Ttrain; % Target binary vector. 
M = size(Xtrain, 2); % Column Number of training sample, input dimension


% Load training set.
load mnist38_validate;

Nv = size(Xvalidate, 1); % Row Number of training sample, input number 
Mv = size(Xvalidate, 2); % Column Number of training sample, input dimension
Xv = [Xvalidate']; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
tv = Tvalidate; % Target binary vector.

% TODO: Hyperparameter lambda which controls the prior on w.
k = [1:20];

load mnist38_test;

Nt = size(Xtest, 1); % Row Number of training sample, input number
Xt = [Xtest']; % Rearrange input转置&include bias term. 产生1行N列的全1数组 截距 for w0 => w0*1
tt = Ttest; % Target binary vector. 
Mt = size(Xtest, 2); % Column Number of training sample, input dimension

figure;   %打开窗口；
axes;     %在当前窗口创建一个包含默认属性的坐标系；
axis([0 21]);  %创建x坐标0～100；y坐标0～1000；
set(gca,'Xtick',0:1:21);   %在x轴上，标记0:20:100的数列；
hold on;

for j=1:20
    
    yv = knn_prediction( X, t, k(j), Xv);
    frac_v = sum((yv > 0.5) - tv == 0)/Nv;
    plot(k(j),frac_v,".g");
    text(k(j),frac_v,['(' num2str(k(j)) ',' num2str(frac_v) ')']);
    hold on;
    yt = knn_prediction( X, t, k(j), Xt);
    frac_t = sum((yt > 0.5) - tt == 0)/Nt;
    plot(k(j),frac_t,".b");
    text(k(j),frac_t,['(' num2str(k(j)) ',' num2str(frac_t) ')']);
    hold on;
    legend("Validation Set","Test Set");
    
end
xlabel("k");
ylabel("Accuracy");
title("k-NN Classification");
hold on;
