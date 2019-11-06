function [Xhat,tensor_cell,rank] = tensor_train(X, eps, varargin)
%TENSOR_TRAIN Summary of this function goes here
%   Detailed explanation goes here
%TENSOR_TRAIN performs tensor train decomposition on the given tensor.
%   This function uses tensor train algorithm with adaptive rank selection
%   to decompose tensor 'X' into a list of small tensors 'tensor_cell'.
% Parameter:
% - 'X' is the tensor for tensor train decomposition.
% - 'eps' is the error tolerance for tensor train decomposition.
% - 'verbose' (optional, default: false) is the switch for covergence
%   information printout.
% Output:
% - 'Xhat' is the approximated tensor of X using tensor train decomposition
% - 'tensor_cell' is a cell array containing all decomposed 3-way tensors: 
%   the first tensor's size is (1, n_1, r_1), the last tensor's size is 
%   (r_{d-1},n_d,1), other i-th tensor's size is (r_{i-1},n_i,r_i).
% - 'rank' is a (d+1) dimension vector containing the resulting rank [r_0,
%   r_1,...,r_{d-1},r_d] with r_0=r_d=1.

    % set defaults for optional inputs
    parser = inputParser;
    parser.addParameter('verbose', false,...
        @(x) isscalar(x) && islogical(x));
    parser.parse(varargin{:});
    verbose = parser.Results.verbose;
    
    % Get number of ways
    nway = ndims(X);
    if nway < 2
        error('The order of input tensor is too low (at least 2).');
    end
    tensor_cell = cell(1, nway);
    dim_array = size(X);
    rank = zeros(nway+1,1);
    rank(1) = 1;
    rank(nway+1) = 1;
    G = double(X);
    PreMul = tensor(ones(1));
    Ykp = X;
    if verbose
        fprintf('Start Tensor Train iterations ... \n');
    end
    for i = 1:nway-1
        if verbose
            fprintf('  %dth tensor procedure:\n',i);
        end
        rp = rank(i);
        nc = dim_array(i);
        [U,S,V] = svd(reshape(G,rp*nc,[]));
        rcmax = min(size(S));
        left = 1; right = rcmax;
        while(left <= right)
            rc = floor((right-left)/2)+left;
            reduce_U = U(:,1:rc);
            reduce_S = S(1:rc,1:rc);
            tt = tensor(reshape(reduce_U*reduce_S,rp,nc,rc));
            G = V(:,1:rc)';
            tMul = ttt(PreMul,tt,ndims(PreMul),1);
            Yk = reshape(ttt(tMul,tensor(G),ndims(tMul),1),dim_array);
            err_F_norm = sqrt(innerprod(Yk-Ykp, Yk-Ykp)*(nway-1));
            
            if verbose
                fprintf('    r%d test: %d, error = %.4e\n',...
                i, rc, err_F_norm);
            end
            
            if err_F_norm >= eps
                left = rc+1;
            else
                if err_F_norm <= eps/10
                    right = rc-1;
                else
                    break
                end
            end
        end
        if verbose
            if err_F_norm < eps
                fprintf('    r%d [success] break at %d, error = %.4e\n',...
                i, rc, err_F_norm);
            else
                fprintf('    r%d [failure] break at %d, error = %.4e\n',...
                i, rc, err_F_norm);
            end
        end
        tensor_cell{i} = tt;
        rank(i+1) = rc;
        PreMul = tMul;
        Ykp = Yk;
    end
    if verbose
        fprintf('  %dth tensor procedure ... ',nway);
    end
    tensor_cell{nway} = tensor(reshape(G,rank(nway),dim_array(nway),1));
    PreMul = ttt(PreMul,tensor_cell{nway},ndims(PreMul),1);
    if verbose
        fprintf('Done.\n');
    end
    Xhat = tensor(reshape(PreMul,dim_array));
end

