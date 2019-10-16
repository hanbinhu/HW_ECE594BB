function [Xhat, lambda, K, K_init] = cp_decomp_als(X, rank)
%CP_DECOMP_ALS performs CP decomposition on the given tensor.
%   This function uses alternating least square (ALS) method to decompose
%   tensor 'X' into two parts: weighting parameter lambda for rank-1
%   vectors, matrices containing all the rank-1 vectors.
% Parameter:
% - 'X' is the tensor for CP decomposition.
% - 'rank' is the rank for CP decomposition.
% Output:
% - 'Xhat' is the approximated tensor of X using CP decomposition.
% - 'lambda' is the weighting parameter for all rank-1 vectors.
% - 'K' is a cell array containing all the rank-1 vectors.
% - 'K_init' is a cell array containing all initial guess of the rank-1
%   vectors.
    maxiters = 1000;
    epstol = 1e-4;

    nway = ndims(X);
    
    % Generate initial guess
    K_init = cell(1, nway);
    for i = 1:nway
        K_init{i} = normc(rand(size(X,i), rank)*2-1);
    end
    
    % Precompute KtK
    KtK = zeros(nway, rank, rank);
    for i = 1:nway
        KtK(i,:,:)=K_init{i}'*K_init{i};
    end
    
    K = K_init;
    lambda = ones(rank,1);
    % ALS steps
    for i = 1:maxiters
        % Iterater for matrices
        for j = 1:nway
            Ahat = double(tenmat(X,j));
            Ahat = Ahat*khatrirao(K{[nway:-1:(j+1),(j-1):-1:1]});
            P=prod(KtK([nway:-1:(j+1),(j-1):-1:1],:,:),1);
            Ahat = Ahat*pinv(squeeze(P));
            % Normalization
            lambda=sqrt(sum(Ahat.^2,1))';
            A = Ahat./lambda';
            % Update
            K{j} = A;
            KtK(j,:,:)= A'*A;
        end
        
        Xhat = tensor(fixsigns(ktensor(lambda, K)));
        
        % Check Convergence
        diff = X-Xhat;
        err = sqrt(innerprod(diff, diff));
        if err < epstol
            break
        end
        fprintf(' Iter %2d: err = %.4e\n', i, err);
    end
end

