clear all;
hold off;

% Load and organize the training/validation/test sets
load mnist38_train
load mnist38_validate
load mnist38_test
Xvalid = Xvalidate; 
Tvalid = Tvalidate; 
N = size(Xtrain, 1);
X = Xtrain'; 
T = Ttrain';
Nv = size(Xvalid, 1);
Xv = Xvalid'; 
Tv = Tvalid';
Nt = size(Xtest, 1); 
Xt = Xtest';
Tt = Ttest';

imagesc(reshape(Xvalid(16,:),[28 28])', [0 1]); colormap('gray');