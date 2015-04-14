clear all;

load mnist38_train;
load mnist38_validate;
load mnist38_test;

% Train a PCA model on the digit training set, keeping all the
% eigenvectors.
[base,mean,projX] = pcaimg(Xtrain', 50);


Nv = size(Xvalidate, 1); %100
Xv = Xvalidate'; 
tv = Tvalidate; 

Xv = Xv-repmat(mean,1,Nv); % substract the mean
projXv=base'*Xv;

dist_v=distmat(projX',projXv');

[m, class] = min(dist_v);
disp(class);

for i=1:Nv
  yv(i)=Ttrain(class(i));
end;
%disp(yv);
%disp(tv);

sum=0;
for i=1:Nv
  sum=sum+(yv(i)==tv(i));
end;

fprintf(1,'frac_v:%4.2f\n',sum/Nv);
