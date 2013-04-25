% Load the toy data set
load_wine;

% Generate random constraints from the groundtruth label. You can also fill
% in Q directly, as long as Q remains symmetric. Note that positive entries
% are must-links, negative entries are cannot-links, and 0 means no
% information. Due to the nature of 2-way partition, the algorithm works
% best when the numbers of ML and CL are approximately balanced.

% Set the number of known labels. Do not set it to 0 because it will cause
% numerical issues for the generalized eigenvalue decomposition. The
% maximum value should be N, the total number of instances.
C=min(10,N);
rp=randperm(N);
tmp=sort(rp(1:C));
clear rp;

Q=zeros(N,N);
for i=1:length(tmp)
    for j=1:length(tmp)
        Q(tmp(i),tmp(j))=label(tmp(i))*label(tmp(j));
    end
end

clear tmp;

% Apply our algorithm
u=csp(L,Q,D_norm,vol,N);

% Turn the relaxed indicator vector into a 2-way partition
result=zeros(N);
result(u>0)=1;
result(u<0)=-1;

% Compute the Rand index
Q_u=result*result';
ri=nnz(Q_u==Q_star)/(N^2);
disp(ri);