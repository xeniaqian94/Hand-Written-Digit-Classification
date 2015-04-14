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

% Command to display the input data
imagesc(reshape(Xtest(3,:),[28 28])', [0 1]); colormap('gray');
fprintf(1,' %d\n', Ttest(3,:)); % 1 for standard output

% Specify the size of our 2-layer MLP
D = size(Xtrain, 2); % number of input units
M = 100; % number of hidden units
K = 1; % number of output units because 3/8 is binary classification

% Initialize connection weights, beta controls the average magnitude
beta = 0.1;
% Weights and biases for layer 1
W1 = beta*randn(D, M);
b1 = zeros(M, 1);
% Weights and biases for layer 2
W2 = beta*randn(M, K);
b2 = zeros(K, 1);

% Learning rate
lambda = 1e-3; 

% Max number of gradient descent iterations
MaxIter = 1000; 

figure;   %打开窗口；
axes;     %在当前窗口创建一个包含默认属性的坐标系；
axis([0 MaxIter*1.2]);  %创建x坐标0～100；y坐标0～1000；
set(gca,'Xtick',0:50:(MaxIter*1.2));   %在x轴上，标记0:20:100的数列；
hold on;
xlabel("No. of Iteration");
ylabel("Average Negative Log Likelihood (Cross Entropy)");
%ylabel("Classification Accuracy");
title("2-Layer MLP with 100 Hidden Units");
hold on;
          
          
for itr = 1:MaxIter
        % Step 1: Forward propagation on training data
  A1 = W1'*X + repmat(b1, 1, N);  % A1-M*N matrix; W1-D*M matrix; X-D*N matrix; b1-M*1 matrix, repmat(b1, 1, N)-M * N matrix
  Z = tanh(A1);  % Compute hidden units. tanh() as the activation function
  A2 = W2'*Z + repmat(b2, 1, N);  % A2-K*N matrix; W2-M*K matrix; Z-M*N matrix
  Y = sigmoid(A2);  % Compute output units. Y-K*N matrix (1*N matrix)
  
  % Compute the average negative log-likelihood (error function) on training data
  train_Err = -mean(T .* log(Y) + (1 - T) .* log(1 - Y));
  
  frac = sum((Y > 0.5) - T == 0)/N;

  % Error Backpropagation
        % Step 2: Evaluate delta_k for all output units
  dEdA2 = Y - T; %K*N matrix
        % Step 3: Evaluate delta_j for all hidden units
  dEdA1 = (1 - Z.^2) .* (W2 * dEdA2); 
  
  % Gradients for weights and biases.
  dEdW2 = Z * dEdA2'; % M*K matrix
  dEdb2 = sum(dEdA2, 2); % sum(x,2) - 以矩阵的每一行为对象，对一行内的数字求和。
 
  dEdW1 = X * dEdA1';
  dEdb1 = sum(dEdA1, 2);
    
  % Forward propagation on validation data
  A1 = W1'*Xv + repmat(b1, 1, Nv);  
  Z = tanh(A1);  % Compute hidden units.
  A2 = W2'*Z + repmat(b2, 1, Nv);  
  Y = sigmoid(A2);  % Compute output units.
  
  % Compute the average negative log-likelihood (error function) on validation data
  valid_Err = -mean(Tv .* log(Y) + (1 - Tv) .* log(1 - Y));
 
  frac_v = sum((Y > 0.5) - Tv == 0)/Nv;
  
  
  % Perform gradient descent
  W1 = W1 - lambda * dEdW1;
  W2 = W2 - lambda * dEdW2;
  b1 = b1 - lambda * dEdb1;
  b2 = b2 - lambda * dEdb2;
  
  plot (itr,train_Err ,".g");
  %plot(itr,frac,".g");
  hold on;
  plot(itr,valid_Err,".b");
  %plot(itr,frac_v,".b");
  hold on;
end
legend("Training Set","Validation Set");
fprintf(1,' Validation Error=%f\n', valid_Err); % 1 for standard output
       
