clear all;

load mnist38_train;
load mnist38_validate;
load mnist38_test;

Nv = size(Xvalidate, 1); %100
tv=Tvalidate;


Nt = size(Xtest, 1); %100
tt=Ttest;

% Train a PCA model on the digit training set, keeping all the
% eigenvectors.

%[base,mean,projX] = pcaimg(Xtrain', 200);
%% Visualize the mean and the first 3 eigendigits.
%figure(1);
%subplot(1,4,1); imagesc(reshape(mean,[28 28])'); colormap('gray'); 
%axis square;
%subplot(1,4,2); imagesc(reshape(base(:,1),[28 28])'); colormap('gray');
%axis square;
%subplot(1,4,3); imagesc(reshape(base(:,2),[28 28])'); colormap('gray');
%axis square;
%subplot(1,4,4); imagesc(reshape(base(:,3),[28 28])'); colormap('gray');
%axis square;

figure;   %打开窗口；
axes;     %在当前窗口创建一个包含默认属性的坐标系；
axis([0 100]);  %创建x坐标0～100；y坐标0～1000；
set(gca,'Xtick',0:5:100);   %在x轴上，标记0:20:100的数列；
hold on;

errorValidation = zeros(1, 10);
errorTest = zeros(1, 10);
numEigenVectors = [2, 3, 5, 10, 20, 40, 50, 70, 80, 90];

for i = 1:10
    K = numEigenVectors(i); 
    
    [base,mean,projX] = pcaimg(Xtrain', K);
    
    Xv = Xvalidate'-repmat(mean,1,Nv); % substract the mean
    Xt = Xtest'-repmat(mean,1,Nt); % substract the mean
    
    projXv=base'*Xv;
    dist_v=distmat(projX',projXv');
    [m, class] = min(dist_v);
    for j=1:Nv
      yv(j)=Ttrain(class(j));
    end;
    sum=0;
    for j=1:Nv
      sum=sum+(yv(j)==tv(j));
    end;
    frac_v=sum/Nv;
    plot(K,frac_v,".g");
    text(K,frac_v,['(' num2str(K) ',' num2str(frac_v) ')']);
    
    projXt=base'*Xt;
    dist_t=distmat(projX',projXt');
    [m, class] = min(dist_t);
    for j=1:Nt
      yt(j)=Ttrain(class(j));
    end;
    sum=0;
    for j=1:Nt
      sum=sum+(yt(j)==tt(j));
    end;
    frac_t=sum/Nt;
    plot(K,frac_t,".b");
    text(K,frac_t,['(' num2str(K) ',' num2str(frac_t) ')']);
    
    hold on;
    legend("Validation Set","Test Set");

end

xlabel("# of Eigen Vectors");
ylabel("Classification Accuracy");
title("PCA classification");
hold on;
