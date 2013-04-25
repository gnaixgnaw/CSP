function U = pareto_multiview (L1, L2)

N = size(L1,1);
[vec,~] = eig(L1,L2);
for i = 1:N
    vec(:,i) = vec(:,i) / norm(vec(:,i));
end

cost1 = diag(vec'*L1*vec);
cost2 = diag(vec'*L2*vec);
[Y,I] = sort(cost1,'ascend');
cost1 = Y;
cost2 = cost2(I);

figure;
hold on;
axis([0 2 0 2]);

scatter(cost1,cost2,'b+');

ex = []; % The cuts to be excluded
U = []; % The output
iter = 0;
pick = 0;
while size(U,2) < 1
    
    % Pick the smallest cut for Graph A, excluding those in ex
    for i=1:N
        if ismember(I(i),ex)==false
            U_ind = i;
            break;
        end
    end
    
    % Compute the Pareto frontier
    cur_cost = cost2(U_ind);
    start = U_ind;
    for i = start:(N-1)
        if cost2(i+1) < cur_cost && ismember(I(i+1),ex)==false
            U_ind = [U_ind, i+1];
            cur_cost = cost2(i+1);
        end
    end
    
    ex = [ex, I(U_ind)']; % Exclude chosen cuts
    if iter > 0 % Skip first pass
%         fprintf('iter:\t%d\n', iter);
        for i=1:size(U_ind,2)
%             if nnz(vec(:,I(U_ind(i)))>0)>0 && nnz(vec(:,I(U_ind(i)))<0)>0
                U = [U, vec(:,I(U_ind(i)))];
                pick = pick + 1;
                fprintf('%d\t%f\t%f\n', pick, cost1(U_ind(i)), cost2(U_ind(i)));
                scatter(cost1(U_ind(i)),cost2(U_ind(i)),'ro');
                text(cost1(U_ind(i)),cost2(U_ind(i)),int2str(pick),'Color',[1 0 0]);
%             end
        end        
    end
    
    iter = iter + 1;
    
end

cost1 = diag(U'*L1*U);
cost2 = diag(U'*L2*U);

hold off;
