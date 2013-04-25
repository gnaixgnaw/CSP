% Constrained Spectral Clustering: The K-way version
%
% Input:
%   The normalized graph Laplacian, L;
%   The constraint matrix, Q;
%   The normalization matrix, D_norm = D^{-1/2};
%   The volume of the graph, vol;
%   The number of clusters, K;
% Ouput:
%   The relaxed cluster indicator vectors, U;

function U = csp_K (L, Q, D_norm, vol, K)

% number of nodes
N = size(L,1);

% set beta such that we have K feasible solutions
lam = svds(Q,2*K);
beta = (lam(K+1)+lam(K))/2-10^(-6);

Q1 = Q - beta*eye(N);

% solve the generalized eigenvalue problem
[vec,~] = eig(L,Q1);

% normalized the eigenvectors
for i = 1:N
    vec(:,i) = vec(:,i)/norm(vec(:,i));
end

% find feasible cuts
satisf = diag(vec'*Q1*vec);
I = find(satisf >= 0);

% sort the feasible cuts by their costs
cost = diag(vec(:,I)'*L*vec(:,I));
[~,ind] = sort(cost,'ascend');

% remove trivial cuts
i = 1;
while 1
    if nnz(vec(:,I(ind(i)))>0)~=0 && nnz(vec(:,I(ind(i)))<0) ~= 0
        break;
    end
    i = i + 1;
end
ind(1:i-1) = [];

% output cluster indicators
ind = ind(1:min(length(ind),K-1));
cost = cost(ind);
U = vec(:,I(ind));
for i = 1:size(U,2)
    U(:,i) = D_norm * (U(:,i) * vol^(1/2)) * (1-cost(i));
end
