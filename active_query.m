% This is our active query strategy, which evaluates the current partition
% u and the currently known queries Q to decide which pair of points to
% query next

% Please see our paper for the technical details of this strategy

function [i,j] = active_query(Q_active,u,N,Q_touch)

[U,S,V]=svd(Q_active);
Q_1=U(:,1)*S(1,1)*V(:,1)';
Q_1(Q_1>1)=1;
Q_1(Q_1<-1)=-1;
P=(Q_1+1)./2;

Q_u=u*u';
Q_u(Q_u>1)=1;
Q_u(Q_u<-1)=-1;
exp_err=(Q_u-1).^2.*P + (Q_u+1).^2.*(1-P);
val=0;
for k=1:N
    for l=(k+1):N
        if (exp_err(k,l)>val) && (Q_touch(k,l)==0)
            i=k;j=l;
            val=exp_err(k,l);
        end
    end
end
