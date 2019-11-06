function ip = ip_Tucker_TN(G_X,K_X,G_Y,K_Y)
%ip_Tucker_TN computes the inner product of two Tucker decomposed tensors.
%   This function takes the Tucker decomposition results of two tensors,
%   and returns the inner product using a tensor network perspective.
% Parameter:
% - 'G_X' is the core tensor 'G' for X from Tucker decomposition.
% - 'K_X' is the matrices cell array 'K' for X from Tucker decomposition.
% - 'G_Y' is the core tensor 'G' for Y from Tucker decomposition.
% - 'K_Y' is the matrices cell array 'K' for Y from Tucker decomposition.
% Output:
% - 'ip' is the resulting inner product of X and Y.
    
    % check rank input
    if numel(K_X) ~= numel(K_Y) || ...
       any(size(G_X) ~= size(G_Y))
        error('Input tensors are not the same size.');
    end
    
    nway = numel(K_X);
    
    % Compute the inner product
    middle_layer = cell(1,nway);
    for i = 1:nway
        middle_layer{i} = ttt(tensor(K_X{i}),tensor(K_Y{i}),1,1);
    end
    
    contract_G_X = G_X;
    for i = 1:nway
        contract_G_X = ttt(contract_G_X,middle_layer{i},1,1);
    end
    
    ip = ttt(contract_G_X,G_Y,1:nway);
end

