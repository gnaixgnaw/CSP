clear all;close all;

%% load data and generate graph

tmp = load('iris.data');

label = tmp(:,end);
data = tmp(:,1:end-1);
clear tmp;

N = size(data,1);

for i = 1:size(data,2)
    data(:,i) = data(:,i) - mean(data(:,i));
end

K = length(unique(label));

my_std = std(data);
my_std(my_std==0) = 1;
data = data*diag(1./my_std);

my_mean = mean(data,1);
my_dist = zeros(N,1);
for i=1:N
    my_dist(i) = norm(data(i,:)-my_mean);
end
sigma = 0.2;
W = eye(N);
for i = 1:N
    for j = (i+1):N
        W(i,j) = exp( -1 * norm(data(i,:)-data(j,:))^2 / (2*(sigma*max(my_dist))^2) );
        W(j,i) = W(i,j);
    end
end

Q_star = zeros(N,N);
for i=1:N
    for j=1:N
        if label(i)==label(j)
            Q_star(i,j) = 1;
        end
    end
end

%% generate random constraints
Omega = eye(N);
idx = zeros((N^2 - N)/2,2);
t = 0;
for i = 1:N
    for j = i+1:N
        t = t+1;
        idx(t,:) = [i,j];
    end
end

tmp = randperm(size(idx,1));
for i = 1:200
    Omega(idx(tmp(i),1),idx(tmp(i),2)) = 1;
    Omega(idx(tmp(i),2),idx(tmp(i),1)) = 1;
end

%% main algorithm

% initialize parameters
alpha = 1e-1;
beta = 1e-1;
mu_start = 10;
mu_final = 1e-1;
iter_max = 5;
iter_in_max = 1000;

% initialize v
X = alpha*W + beta*(Q_star.*Omega);
X = (X+X')/2;
[~,~,v] = svds(X,K);

% do the main thing
[Q,~,~] = sf(W,v,K,Q_star,Omega,alpha,beta,mu_start,mu_final,iter_max,iter_in_max);
[~,~,v] = svds(Q,K);

% post-processing, generate clusters
clust = kmeans(v,K,'EmptyAction','singleton','Replicates',10);
[ari,ri] = RandIndex(clust,label);
fprintf('Adjusted RandIndex:\t%f\tRandIndex:\t%f\n', ari, ri);
