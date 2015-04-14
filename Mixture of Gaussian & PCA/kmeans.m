% Performs clustering using the k-means algorithm.
%
% Input:
%
%   x(:,t) = the N-dimensional training vector for the tth training case
%   K = number of clusters
%   iters = number of iterations of EM to apply
%
% Output:
%
%   means(:, c) = mean of the cth cluster


function [means] = kmeans(x, K, iters)
    % Initialize the means to K random training vectors
    perm = randperm(size(x, 2)); %产生1到N的整数的无重复的随机排列
    means = x(:, perm(1:K)); %随机选K个作为mean
    
    N = size(x, 2);
    for iter=1:iters
        dist = nan(K, N);  %initialize an K*N matrix with NaN
        for k=1:K
            dist(k, :) = distmat(x', means(:, k)');
        end
        [m, class] = min(dist);
        for k=1:K
            means(:,k) = mean(x(:, class == k), 2);  %求每一行的均值
        end
    end
