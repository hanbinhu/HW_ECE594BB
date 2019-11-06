function ip = ip_TT_CP_TN(TT_X,TT_rank_X,l_Y,K_Y)
%ip_TT_CP_TN computes the inner product of two decomposed tensors.
%   This function takes one Tensor Train decomposition of one tensor 'X'
%   and the CP decomposition of another tensor 'Y', and returns the inner
%   product using a tensor network perspective.
% Parameter:
% - 'TT_X' is the a cell array containing all the tensors for X from Tensor
%   Train decomposition.
% - 'TT_rank_X' is a vector containing all the tensor ranks for X from
%   Tensor Train decomposition.
% - 'l_Y' is the weighting parameter 'lambda' for Y from CP decomposition.
% - 'K_Y' is the matrices cell array 'K' for Y from CP decomposition.
% Output:
% - 'ip' is the resulting inner product of X and Y.
ip = 0;
end

