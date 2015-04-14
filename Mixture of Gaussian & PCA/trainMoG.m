clear all;

load mnist38_train;

N = size(Xtrain, 1); %200 in the training data, 前100个是3 后100个是8
t=Ttrain;

Xtrain_3=Xtrain(Ttrain==0,:);
[p_3,mu_3,vary_3,logProbX_3] = mogEM(Xtrain_3',2,40);

Xtrain_8=Xtrain(Ttrain==1,:);
[p_8,mu_8,vary_8,logProbX_8] = mogEM(Xtrain_8',2,40);

% Visualize the mean and the variance as images of 3
subplot(2,4,1); imagesc(reshape(mu_3(:,1),[28 28])'); colormap('gray'); 
title("mean(1)");
text(0,-5,['mixing proportion: ' num2str(p_3(1)) ':' num2str(p_3(2)) ';    final log-likelihood: ' num2str(logProbX_3(40)) ' ']);
axis square;
subplot(2,4,2); imagesc(reshape(mu_3(:,2),[28 28])'); colormap('gray');
title("mean(2)");
axis square;
subplot(2,4,3); imagesc(reshape(vary_3(:,1),[28 28])'); colormap('gray');
title("vary(1)");
axis square;
subplot(2,4,4); imagesc(reshape(vary_3(:,2),[28 28])'); colormap('gray');
title("vary(2)");
axis square;

    %fprintf(1,"%4.2f\n",sum(mogLogProb(p_3,mu_3,vary_3,Xtrain_3')));

% Visualize the mean and the variance as images of 8
subplot(2,4,5); imagesc(reshape(mu_8(:,1),[28 28])'); colormap('gray'); 
title("mean(1)");
text(0,-5,['mixing proportion: ' num2str(p_8(1)) ':' num2str(p_8(2)) ';    final log-likelihood: ' num2str(logProbX_8(40)) ' ']);
axis square;
subplot(2,4,6); imagesc(reshape(mu_8(:,2),[28 28])'); colormap('gray');
title("mean(2)");
axis square;
subplot(2,4,7); imagesc(reshape(vary_8(:,1),[28 28])'); colormap('gray');
title("vary(1)");
axis square;
subplot(2,4,8); imagesc(reshape(vary_8(:,2),[28 28])'); colormap('gray');
title("vary(2)");
axis square;
