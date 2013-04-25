% Load the data set, trim it into two classes.
tmp = load('wine.data');
label = tmp(:,1);
data = tmp(label~=1,2:end);
label(label==1) = [];
clear tmp;

% Convert labels, '1' for one class and '-1' for the other.
label(label==2) = 1;
label(label==3) = -1;

% The complete constraint matrix
Q_star = label*label';

% Center and normalize the data attributes. Each row is an instace and
% each column is an attribute.
for i = 1:size(data,2)
    data(:,i) = data(:,i) - mean(data(:,i));
end
my_var = var(data);
% The size of the data set
N = size(data,1);

% Compute the similarity matrix A using RBF kernel.
% Set diagonal entries to 0 for numerical consideration.
A = zeros(N,N);
for i = 1:N
    for j=(i+1):N
        A(i,j) = exp(-1*sum((data(i,:)-data(j,:)).^2./(2*my_var)));
        A(j,i) = A(i,j);
    end
end

% Compute the graph Laplacian.
D = diag(sum(A)); vol = sum(diag(D)); D_norm = D^(-1/2);
L = eye(N) - D_norm*A*D_norm;

% Construct View 1 for multi-view learning
data1 = data(:,1:6);
my_var1 = var(data1);
A1 = zeros(N,N);
for i = 1:N
    for j = (i+1):N
        A1(i,j) = exp(-1*sum((data1(i,:)-data1(j,:)).^2./(2*my_var1)));
        A1(j,i) = A1(i,j);
    end
end
D1 = diag(sum(A1)); vol1 = sum(diag(D1)); D_norm1 = D1^(-1/2);
L1 = eye(N) - D_norm1*A1*D_norm1;

% Construct View 2 for multi-view learning
data2 = data(:,7:end);
my_var2 = var(data2);
A2 = zeros(N,N);
for i = 1:N
    for j = (i+1):N
        A2(i,j) = exp(-1*sum((data2(i,:)-data2(j,:)).^2./(2*my_var2)));
        A2(j,i) = A2(i,j);
    end
end
D2 = diag(sum(A2)); vol2 = sum(diag(D2)); D_norm2 = D2^(-1/2);
L2 = eye(N) - D_norm2*A2*D_norm2;
