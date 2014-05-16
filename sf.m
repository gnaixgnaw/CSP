% Input:
% W: graph affinity matrix, NxN
% v: initial cut, NxK
% K: number of clusters
% Q_star: ground truth constraint matrix, NxN
% Omega: mask (what entries of Q_star are revealed), NxN
%
% Output:
% Q: learned constraint matrix, NxN
% v: final constrained cut, NxK
% obj_overall: objective value

function [Q,v,obj_overall] = sf_core(W,v,K,Q_star,Omega,alpha,beta,mu_start,mu_final,iter_max,iter_in_max)

tau = 1.99;
eta_mu = 1/2;
N = size(W,1);

for iter = 1:iter_max
    mu = mu_start;
    if iter > 1
        Q_last = Q;
    end
    Q = zeros(size(Q_star));
    obj_overall = [];
    while 1
        for iter_in = 1:iter_in_max
            Y = Q - tau*((Q-Q_star).*Omega) + tau*beta*(v*v');
            [U,S,V] = svd(Y);
            for i=1:N
                S(i,i) = max(S(i,i) - tau*mu, 0);
            end
            Q = U*S*V';            
            obj =  mu*sum(diag(S))+norm((Q-Q_star).*Omega,'fro')^2/2 - beta*trace(v'*Q*v);
            obj_overall = [obj_overall,-alpha*trace(v'*W*v)+ mu_final*sum(diag(S)) + norm((Q-Q_star).*Omega,'fro')^2/2 - beta*trace(v'*Q*v)];
            if iter_in > 1
                diff_obj = abs(obj_overall(end-1) - obj_overall(end))/abs(obj_overall(end-1));
                if diff_obj<1e-5
                    break
                end
            end
        end

        if mu == mu_final
            break;
        else
            mu = max(mu*eta_mu,mu_final);
        end
    end
    
    X = alpha*W + beta*(Q_star.*Omega);
    X = (X+X')/2;
    [~,~,v] = svds(X,K);

    if iter > 1 
        diff_Q = norm(Q-Q_last,'fro')/max(1,norm(Q_last,'fro'));
        if diff_Q < 1e-2
            break;
        end
    end
end