function [Xhat,lambda,K,K_init,converge] = cp_decomp_als(X, rank, varargin)
%CP_DECOMP_ALS performs CP decomposition on the given tensor.
%   This function uses alternating least square (ALS) method to decompose
%   tensor 'X' into two parts: weighting parameter lambda for rank-1
%   vectors, matrices containing all the rank-1 vectors.
% Parameter:
% - 'X' is the tensor for CP decomposition.
% - 'rank' is the rank for CP decomposition.
% - 'maxiters' (optional, default: 100) is the maximum iteration for the CP 
%   decomposition.
% - 'difftol' (optional, default: 1e-4) is the tolerance value for the
%   change of normalized error norm.
% - 'abstol' (optional, default: 1e-2) is the tolerance value for the
%   normalized error norm.
% - 'verbose' (optional, default: false) is the switch for covergence
%   information printout.
% Output:
% - 'Xhat' is the approximated tensor of X using CP decomposition.
% - 'lambda' is the weighting parameter for all rank-1 vectors.
% - 'K' is a cell array containing all the rank-1 vectors.
% - 'K_init' is a cell array containing all initial guess of the rank-1
%   vectors.
% - 'converge' gives if the CP decomposition coverges.

    % set defaults for optional inputs
    parser = inputParser;
    parser.addParameter('maxiters', 100,...
        @(x) isscalar(x) && x>0 && floor(x)==x);
    parser.addParameter('difftol', 1e-4,...
        @(x) isscalar(x) && x>0);
    parser.addParameter('abstol', 1e-2,...
        @(x) isscalar(x) && x>0);
    parser.addParameter('verbose', false,...
        @(x) isscalar(x) && islogical(x));
    parser.parse(varargin{:});
    maxiters = parser.Results.maxiters;
    difftol = parser.Results.difftol;
    abstol = parser.Results.abstol;
    verbose = parser.Results.verbose;
    
    % Get number of ways
    nway = ndims(X);
    
    % Generate initial guess
    if verbose
        fprintf('Generating initial guess ... ');
    end
    K_init = cell(1, nway);
    for i = 1:nway
        K_init{i} = normc(rand(size(X,i), rank)*2-1);
    end
    if verbose
        fprintf('Done\n');
    end
    
    % Precompute KtK
    if verbose
        fprintf('Precomputing ... ');
    end
    KtK = zeros(nway, rank, rank);
    for i = 1:nway
        KtK(i,:,:)=K_init{i}'*K_init{i};
    end
    if verbose
        fprintf('Done\n');
    end
    
    K = K_init;
    lambda = ones(rank,1);
    % ALS steps
    ipX = sqrt(innerprod(X, X));
    perr = 0;
    converge = false;
    if verbose
        fprintf('Start ALS iterations ... ');
    end
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
        nerr = err / ipX;
        err_diff = abs(nerr-perr);
        if verbose
            fprintf(' Iter %2d: norm_error = %.4e, diff_error = %.4e\n',...
                i, nerr, err_diff);
        end
        if err_diff < difftol && nerr < abstol
            converge = true;
            break;
        end
        perr = nerr;
    end
    if verbose
        if ~converge
            warning(['The CP decomposition is not converged under the '...
                sprintf('maximum %d iterations with the ', maxiters)...
                sprintf('given tolerance of %.1e (abs)/%.1e (diff).\n',...
                abstol, difftol)]);
        else
            fprintf('Coverged.\n');
        end
    end
    
end

