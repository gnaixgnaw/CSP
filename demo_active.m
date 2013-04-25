% This is the demo program for active spectral clustering

clear;

% Load the data set
load_wine;
% load_iris;

% T is how many times we repeat the randomized query strategy, so that we
% can compare it to our active query strategy
T=10;

% Number of queries
max_iter=min(N*(N-1)/2,2*N);

disp('active method starts');

% We keep track of the performance of the active algorithm at each
% iteration, i.e. after each new query
record_active=zeros(max_iter,1);

% Initiate the active algorithm by computing the unconstrained spectral
% clustering
[vec,val]=svd(L);
v=vec(:,N-1)/norm(vec(:,N-1))*vol^(1/2);
u_active=D_norm*v;

% Q_active keeps track of the answers returned from the oracle
Q_active=eye(N);

% Q_touch keeps track of what has been queried and what has not
Q_touch=eye(N);

ind=1;
while ind<=max_iter

    % Evaluate the performance of the current result, using Rand index
    record_active(ind) = eval_rand(u_active,Q_star);

    % Compute which pair of points to query next
    [i,j] = active_query(Q_active,u_active,N,Q_touch);
    
    % Query the oracle
    Q_active(i,j)=Q_star(i,j);
    Q_active(j,i)=Q_star(j,i);
    Q_touch(i,j)=1;
    Q_touch(j,i)=1;

    % Update the partition
    u_active = csp(L,Q_active,D_norm,vol,N);

    ind=ind+1;

end

% This is the baseline method where we randomly choose pairs of points to
% query and compute the partition using the same constrained
% spectralclustering algorithm csp()
disp('random method starts');

record_random=zeros(max_iter,T);

for out_iter=1:T

    [vec,val]=svd(L);
    v=vec(:,N-1)/norm(vec(:,N-1))*vol^(1/2);
    u_random=D_norm*v;

    Q_random=eye(N);
    Q_touch=eye(N);

    ind=1;
    while ind<=max_iter

        record_random(ind,out_iter) = eval_rand(u_random,Q_star);
        [I,J]=find(Q_random==0);
        tmp=randi(length(I));
        i=I(tmp);
        j=J(tmp);
        Q_random(i,j)=Q_star(i,j);
        Q_random(j,i)=Q_star(j,i);

        u_random = csp(L,Q_random,D_norm,vol,N);

        ind=ind+1;

    end
end

% Plotting the results

figure;
set(gca,'fontsize',12);
xlim([0 max_iter]);
xlabel('# constraints queried','fontsize',12);
ylabel('Rand index','fontsize',12);
hold on;
plot(record_active,'b','Linewidth',2);
plot(max(record_random,[],2),'-.r','Linewidth',1);
plot(mean(record_random,2),'r','Linewidth',1);
plot(min(record_random,[],2),':r','Linewidth',1);
legend('active','random-max','random-avg','random-min');
hold off;
