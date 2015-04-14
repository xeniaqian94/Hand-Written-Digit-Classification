hold off;
clear all;
load mnist38_train;
load mnist38_validate; 
load mnist38_test;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];
%numComponent=[2]; % for testing

N = size(Xtrain, 1); %200 in the training data, 前100个是3 后100个是8
t = Ttrain;
Xtrain_3=Xtrain(Ttrain==0,:);
Xtrain_8=Xtrain(Ttrain==1,:);

Nv = size(Xvalidate, 1); 
tv = Tvalidate;

Nt = size(Xtest, 1); 
tt = Ttest;


figure;   %打开窗口；
axes;     %在当前窗口创建一个包含默认属性的坐标系；
axis([0 30]);  %创建x坐标0～100；y坐标0～1000；
set(gca,'Xtick',0:5:30);   %在x轴上，标记0:20:100的数列；
hold on;

train_iter=10;
model_iter=10;

for i=1:4
%for i = 1 : 1
   K = numComponent(i);
% Train a MoG model with K components for digit 3 
% You should repeat a few times and pick the model with the best final 
% log probability.  
%-------------------- Add your code here --------------------------------
   max_log=zeros(model_iter,1);
   for j=1:train_iter
      [this_p,this_mu,this_vary,this_log] = mogEM(Xtrain_3',K,model_iter);
      if (this_log(model_iter)>max_log(model_iter))
         p_3=this_p; mu_3=this_mu;vary_3=this_vary; logProbX_3=this_log;
         max_log=this_log;
      end;
   end;

% Train a MoG model with K components for digit 8
% You should repeat a few times and pick the model with the best final 
% log probability.  
%-------------------- Add your code here --------------------------------
   max_log=zeros(model_iter,1);
   for j=1:train_iter
      [this_p,this_mu,this_vary,this_log] = mogEM(Xtrain_8',K,model_iter);
      if (this_log(model_iter)>max_log(model_iter))
         p_8=this_p; mu_8=this_mu;vary_8=this_vary; logProbX_8=this_log;
         max_log=this_log;
      end;
   end;


% Caculate the probability P(Digit=3|Image) and P(Digit=8|Image), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function

   %classify on training set
   logProb_3 = mogLogProb(p_3,mu_3,vary_3,Xtrain');
   [logPost_3, L] = normalizeLogspace(logProb_3-log(0.5)*ones(1,N));
%   [logPost_3, L] = normalizeLogspace(logProb_3);
   post_3 = exp(logPost_3);
   
   logProb_8 = mogLogProb(p_8,mu_8,vary_8,Xtrain');
   [logPost_8, L] = normalizeLogspace(logProb_8-log(0.5)*ones(1,N));
%   [logPost_8, L] = normalizeLogspace(logProb_8);
   post_8 = exp(logPost_8);
   
   y=(post_8>post_3);
   frac(1,i)=nnz(y'==t)/N;
   
%   fprintf(1,"on training set, ");
%   disp(frac);
   
%   plot(K,frac,".m");
   text(K,frac(1,i),['(' num2str(K) ',' num2str(frac(1,i)) ')']);
    
   hold on;
   
   %classify on validation set
   logProb_3 = mogLogProb(p_3,mu_3,vary_3,Xvalidate');
   [logPost_3, L] = normalizeLogspace(logProb_3-log(0.5)*ones(1,Nv));
%   [logPost_3, L] = normalizeLogspace(logProb_3);
   post_3 = exp(logPost_3);
   
   logProb_8 = mogLogProb(p_8,mu_8,vary_8,Xvalidate');
   [logPost_8, L] = normalizeLogspace(logProb_8-log(0.5)*ones(1,Nv));
%   [logPost_8, L] = normalizeLogspace(logProb_8);
   post_8 = exp(logPost_8);
   
   yv=(post_8>post_3);
   frac_v(1,i)=nnz(yv'==tv)/Nv;
   
%   plot(K,frac_v,".b");
   text(K,frac_v(1,i),['(' num2str(K) ',' num2str(frac_v(1,i)) ')']);
    
   hold on;
   
   %classify on test set
   logProb_3 = mogLogProb(p_3,mu_3,vary_3,Xtest');
   [logPost_3, L] = normalizeLogspace(logProb_3-log(0.5)*ones(1,Nt));
%   [logPost_3, L] = normalizeLogspace(logProb_3);
   post_3 = exp(logPost_3);
   
   logProb_8 = mogLogProb(p_8,mu_8,vary_8,Xtest');
   [logPost_8, L] = normalizeLogspace(logProb_8-log(0.5)*ones(1,Nt));
%   [logPost_8, L] = normalizeLogspace(logProb_8);
   post_8 = exp(logPost_8);
   
   yt=(post_8>post_3);
   frac_t(1,i)=nnz(yt'==tt)/Nt;
   
%   fprintf(1,"on test set, ");
%   disp(frac_t);
   
%   plot(K,frac_t,".g");
   text(K,frac_t(1,i),['(' num2str(K) ',' num2str(frac_t(1,i)) ')']);
    
   hold on;
   
   
%-------------------- Add your code here --------------------------------
end

grid on;
plot(numComponent, frac,'m',numComponent, frac_v, 'b',numComponent, frac_t, 'g' );
xlabel("# of Gaussian Components");
ylabel("Classification Accuracy");
title("MoG classification");
legend("Training Set","Validation Set","Test Set");
hold on;

