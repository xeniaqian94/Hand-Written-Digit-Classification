clear all;

load mnist38_train;

N = size(Xtrain, 1); %200 in the training data, 前100个是3 后100个是8
t=Ttrain;

Xtrain_3=Xtrain(Ttrain==0,:);
[p,mu,vary,logProbX] = mogEM(Xtrain_3',2,20);

% Visualize the mean and the first 3 eigendigits.
figure(1);
subplot(1,4,1); imagesc(reshape(mu(:,1),[28 28])'); colormap('gray'); 
axis square;
subplot(1,4,2); imagesc(reshape(mu(:,2),[28 28])'); colormap('gray');
axis square;
subplot(1,4,3); imagesc(reshape(vary(:,1),[28 28])'); colormap('gray');
axis square;
subplot(1,4,4); imagesc(reshape(vary(:,2),[28 28])'); colormap('gray');
axis square;