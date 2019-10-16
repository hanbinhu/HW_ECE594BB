function [X, lambda, K] = generate_low_rank_tensor(dim, rank, lambda_range)
%GENERATE_LOW_RANK_TENSOR generates a random low-rank tensor.
%   The tensor is generated in a Kruskal format. The number of ways of the
%   generated tensor is given by the length of 'dim'. The kruskal format
%   uses the normalized matrices and a weighting array lambda.
% Parameter:
% - 'dim' is an array describing the size of each way. 
% - 'rank' stands for the rank of CP decomposition.
% - 'lambda_range' gives the range of lambda following a uniform
%   distribution [-lambda_range, lambda_range]
% Output:
% - 'X' is the generated tensor.
% - 'lambda' is the weighting array lambda.
% - 'K' gives the normalized matrices.
    nway=length(dim);
    K=cell(1,nway);
    for i=1:nway
        K{i}=normc(rand(dim(i), rank)*2-1);
    end
    lambda=(rand(rank,1)*2-1)*lambda_range;
    X=tensor(fixsigns(ktensor(lambda,K)));
end