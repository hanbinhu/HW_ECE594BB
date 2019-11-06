function ip = ip_TT_CP_TN(TT_X,l_Y,K_Y)
%ip_TT_CP_TN computes the inner product of two decomposed tensors.
%   This function takes one Tensor Train decomposition of one tensor 'X'
%   and the CP decomposition of another tensor 'Y', and returns the inner
%   product using a tensor network perspective.
% Parameter:
% - 'TT_X' is the a cell array containing all the tensors for X from Tensor
%   Train decomposition.
% - 'l_Y' is the weighting parameter 'lambda' for Y from CP decomposition.
% - 'K_Y' is the matrices cell array 'K' for Y from CP decomposition.
% Output:
% - 'ip' is the resulting inner product of X and Y.

    % check rank input
    if numel(TT_X) ~= numel(K_Y)
        error('Input tensors are not the same size.');
    end
    
    nway = numel(K_Y);
    
    % Compute the inner product
    middle_layer = cell(1,nway);
    for i = 1:nway-1
        middle_layer{i} = permute(ttt(TT_X{i},tensor(K_Y{i}),2,1),[3,1,2]);
    end
    middle_layer{1} = squeeze(middle_layer{1});
    middle_layer{nway} = ...
        permute(ttt(TT_X{nway},tensor(K_Y{nway}),2,1),[2,1]);
    
    ip = 0;
    for i = 1:numel(l_Y)
        w = double(middle_layer{1}(i,:))';
        for j = 2:nway-1
            w = w*double(squeeze(middle_layer{j}(i,:,:)));
        end
        w = w*double(middle_layer{nway}(i,:));
        ip = ip+l_Y(i)*w;
    end
end

