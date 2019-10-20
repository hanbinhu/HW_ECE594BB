function [Xhat,G,K,K_init,converge] = tucker_hooi(X, rank, varargin)
%TUCKER_HOOI performs Tucker decomposition on the given tensor.
%   This function uses High-Order Orthogonal Iteration (HOOI) method to
%   decompose tensor 'X' into two parts: a core tensor G and multiple
%   orthogonal matrices K.
% Parameter:
% - 'X' is the tensor for Tucker decomposition.
% - 'rank' is the multilinear rank for Tucker decomposition.
% - 'maxiters' (optional, default: 20) is the maximum iteration for the 
%   Tucker decomposition.
% - 'difftol' (optional, default: 1e-4) is the tolerance value for the
%   change of normalized error norm.
% - 'abstol' (optional, default: 1e-2) is the tolerance value for the
%   normalized error norm.
% - 'verbose' (optional, default: false) is the switch for covergence
%   information printout.
% Output:
% - 'Xhat' is the approximated tensor of X using Tucker decomposition.
% - 'G' is the core tensor of Tucker decomposition.
% - 'K' is a cell array containing all the orthogonal matrices.
% - 'K_init' is an initial cell array with all HOSVD's matrices results.
% - 'converge' gives if the Tucker (HOOI) decomposition coverges.
    
    % set defaults for optional inputs
    parser = inputParser;
    parser.addParameter('maxiters', 20,...
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
    
    % check rank input
    if any(~isnumeric(rank)) ||...
       length(rank) ~= nway ||...
       any(floor(rank) ~= rank)
        error('Rank is not a valid input argument.');
    end
    for i = 1:nway
        if rank(i) > size(X,i)
            error('Rank is not a valid input argument.');
        end
    end
    
    % Generate initial guess
    if verbose
        fprintf('Generate Initial Results via HOSVD.\n');
    end
    [~,G,K_init] = tucker_hosvd(X, rank, 'verbose', verbose);
    
    K = K_init;
    % HOOI steps
    ipX = sqrt(innerprod(X, X));
    perr = 0;
    converge = false;
    if verbose
        fprintf('Start HOOI iterations ...\n');
    end
    for i = 1:maxiters
        % Iterater for matrices
        G = X;
        for j = 1:nway
            if j < nway
                Y = ttm(G,K,(j+1:nway),'t');
            else
                Y = G;
            end
            Yj = double(tenmat(Y,j));
            [U,~,~] = svd(Yj);
            K{j} = U(:,1:rank(j));
            G = ttm(G,K,j,'t');
        end
        
        % Check Convergence
        normG = sqrt(innerprod(G,G));
        nerr = abs((ipX-normG)/ipX);
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
            warning(['The Tucker decomposition is not converged under '...
                sprintf('the maximum %d iterations with the ', maxiters)...
                sprintf('given tolerance of %.1e (abs)/%.1e (diff).\n',...
                abstol, difftol)]);
        else
            fprintf('Coverged.\n');
        end
    end
    Xhat = tensor(ttensor(G, K));
end

