function ip = ip_TT_Tucker_TN(TT_X,G_Y,K_Y)
%ip_TT_Tucker_TN computes the inner product of two decomposed tensors.
%   This function takes one Tensor Train decomposition of one tensor 'X'
%   and the Tucker decomposition of another tensor 'Y', and returns the
%   inner product using a tensor network perspective.
% Parameter:
% - 'TT_X' is the a cell array containing all the tensors for X from Tensor
%   Train decomposition.
% - 'G_Y' is the core tensor 'G' for Y from Tucker decomposition.
% - 'K_Y' is the matrices cell array 'K' for Y from Tucker decomposition.
% Output:
% - 'ip' is the resulting inner product of X and Y.
    
    % check rank input
    if numel(TT_X) ~= numel(K_Y)
        error('Input tensors are not the same size.');
    end
    
    nway = numel(K_Y);
    
    % Compute the inner product
    XX=tensor(1);
    for i = 1:nway-1
        temp = permute(ttt(TT_X{i},tensor(K_Y{i}),2,1),[1,3,2]);
        XX=ttt(XX,temp,ndims(XX),1);
    end
    XX=squeeze(ttt(XX,ttt(TT_X{nway},tensor(K_Y{nway}),2,1),ndims(XX),1));
    ip=innerprod(XX,G_Y);
end