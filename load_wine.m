% Load the data set, trim it into two classes.
tmp=load('wine.data');

% Create labels, '1' for one class and '-1' for the other.
label=tmp(:,1);
label(label==2)=1;
label(label==3)=-1;

% Center and normalize the data attributes. Each row is an instace and
% each column is an attribute.
data=tmp(:,2:end);
clear tmp;
for i=1:size(data,2)
    data(:,i)=data(:,i)-mean(data(:,i));
end
my_var=var(data);
% The size of the data set
N=size(data,1);

% Compute the similarity matrix A using RBF kernel.
A=zeros(N,N);
for i=1:N
    for j=1:N
        A(i,j)=exp(-1*sum((data(i,:)-data(j,:)).^2./(2*my_var)));
    end
end

% Set diagonal entries to 0 for numerical consideration.
for i=1:N
    A(i,i)=0;
end

% Compute the graph Laplacian.
D=diag(sum(A));
vol=sum(diag(D));
D_norm=D^(-1/2);
L=eye(N)-D_norm*A*D_norm;

% The complete constraint matrix
Q_star=label*label';
