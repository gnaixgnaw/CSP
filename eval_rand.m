% Evaluate the quality of u using Rand index
% Q_star is the grountruth result
function ri = eval_rand(u,Q_star)

result=zeros(size(u));
result(u>0)=1;
result(u<0)=-1;
Q_u=result*result';

ri=nnz(Q_u==Q_star)/(size(Q_star,1))^2;