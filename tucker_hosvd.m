function [Xhat,G,K] = tucker_hosvd(X, rank, varargin)
%TUCKER_HOSVD performs Tucker decomposition on the given tensor.
%   This function uses High-Order SVD (HOSVD) method to decompose
%   tensor 'X' into two parts: a core tensor G and multiple orthogonal
%   matrices K.
% Parameter:
% - 'X' is the tensor for Tucker decomposition.
% - 'rank' is the multilinear rank for Tucker decomposition.
% - 'verbose' (optional, default: false) is the switch for covergence
%   information printout.
% Output:
% - 'Xhat' is the approximated tensor of X using Tucker decomposition.
% - 'G' is the core tensor of Tucker decomposition.
% - 'K' is a cell array containing all the orthogonal matrices.
    
    % set defaults for optional inputs
    parser = inputParser;
    parser.addParameter('verbose', false,...
        @(x) isscalar(x) && islogical(x));
    parser.parse(varargin{:});
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
        
    % HOSVD
    if verbose
        fprintf('Starting HOSVD computing ... ');
    end
    K = cell(1, nway);
    G = X;
    for i = 1:nway
        Xi = double(tenmat(X,i));
        [U,~,~] = svd(Xi);
        K{i} = U(:,rank(i));
        G = ttm(G, K{i}, i, 't');
    end
    Xhat = tensor(ttensor(G, K));
    if verbose
        fprintf('Done.\n');
    end
        
    % Check Convergence
    if verbose
        ipX = sqrt(innerprod(X, X));
        diff = X-Xhat;
        err = sqrt(innerprod(diff, diff));
        nerr = err / ipX;
        fprintf('HOSVD Results: abs_error = %.4e, norm_error = %.4e\n',...
            err, nerr);
    end
end

