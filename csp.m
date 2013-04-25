% Input: graph Laplacian L, constraint matrix Q, the normalization matrix
% D_norm, the graph of the graph vol, the number of instances N
% Output: the relaxed cluster indicator vector u (for 2-way partition only)
function u = csp (L,Q,D_norm,vol,N)

% Normalize the constraint matrix
Q_norm = D_norm*Q*D_norm;

% Set the parameter alpha to 0.5*(the largest eigenvalue of Q_norm)
lam=svds(Q_norm,1);
Q1=Q_norm-(lam*0.5)*eye(N);
[vec,val] = eig(L,Q1);

% Find the positive eigenvectors
I=find(diag(val)>=0);

% Compute the respective costs of the cuts
cost=zeros(length(I),1);
for i=1:length(I)
    v=vec(:,I(i))/norm(vec(:,I(i)))*vol^(1/2);
    cost(i)=v'*L*v;
end

% Find the one with minimum cost
[cost_val,cost_ind]=sort(cost,'ascend');
i=1;
while i<=length(cost)
    % Deal with numerical issues here
    if cost_val(i)>10^(-10)
        ind = cost_ind(i);
        break;
    end
    i=i+1;
end

% Output the cluster indicator vector
v=vec(:,I(ind))/norm(vec(:,I(ind)))*vol^(1/2);
u=D_norm*v;
