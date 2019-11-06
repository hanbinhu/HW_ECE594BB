function ip = ip_CP_TN(l_X,K_X,l_Y,K_Y)
%ip_CP_TN computes the inner product of two CP decomposed tensors.
%   This function takes the CP decomposition results of two tensors, and
%   returns the inner product using a tensor network perspective.
% Parameter:
% - 'l_X' is the weighting vector 'lambda' for X from CP decomposition.
% - 'K_X' is the matrices cell array 'K' for X from CP decomposition.
% - 'l_Y' is the weighting vector 'lambda' for Y from CP decomposition.
% - 'K_Y' is the matrices cell array 'K' for Y from CP decomposition.
% Output:
% - 'ip' is the resulting inner product of X and Y.
    
    % check rank input
    if numel(K_X) ~= numel(K_Y) || numel(l_X) ~= numel(l_Y)
        error('Input tensors are not the same size.');
    end
    
    nway = numel(K_X);
    
    % Compute the inner product
    w = l_X*l_Y';
    for i = 1:nway
        w = w.*(K_X{i}'*K_Y{i});
    end
    ip = sum(sum(double(w)));
end

