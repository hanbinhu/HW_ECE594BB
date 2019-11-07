function [X, G, K] = generate_low_rank_tensor(dim,rank,range,form,param)
%GENERATE_LOW_RANK_TENSOR generates a random low-rank tensor.
%   The tensor is generated in a Kruskal format or Tucker format. The
%   number of ways of the generated tensor is given by the length of 'dim'.
%   - The Kruskal format uses the normalized matrices and a weighting array
%   lambda.
%   - The Tucker format uses the orthogonal matrices and a weighting array
%   lambda.
% Parameter:
% - 'dim' is an array describing the size of each way. 
% - 'rank' stands for the rank of Kruskal form or multilinear rank in
%   Tucker form.
% - 'range' gives the range of lambda or core tensor following a uniform
%   distribution [-range, range]
% - 'form' gives which format is used. ('Kruskal', 'Tucker')
% Output:
% - 'X' is the generated tensor.
% - 'G' is the weighting array lambda in Kruskal form or core tensor in
%   Tucker form.
% - 'K' gives the normalized matrices in Kruskal form or orthogonal
%   matrices in Tucker form.
    nway=length(dim);
    if strcmp(form, 'Kruskal')
        K=cell(1,nway);
        for i=1:nway
            K{i}=normc(rand(dim(i), rank)*2-1);
        end
        G=(rand(rank,1)*2-1)*range;
        X=tensor(ktensor(G,K));
    else
        if strcmp(form, 'Tucker')
            if numel(param) < 1
                error('Require argument for Tucker problem generation.')
            end
            K=cell(1,nway);
            for i=1:nway
                R=randn(dim(i), rank(i));
                R(:,1) = R(:,1)/norm(R(:,1));
                for j = 2:rank(i)
                    U = R(:,j);
                    for k = 1:j-1
                        U = U - (U'*R(:,k))/(R(:,k)'*R(:,k))*R(:,k);
                    end
                    R(:,j) = U/norm(U);
                end
                K{i}=R;
            end
            rand_tensor=rand(rank)*2-1;
            weight=ones(rank);
            distmax = sum(rank)-nway;
            for i = 1:numel(weight)
                dist = 0;
                k = i-1;
                for j = nway:-1:1
                    base = prod(rank(1:j-1));
                    index = floor(k/base);
                    k=mod(k,base);
                	dist = dist+index;
                end
                weight(i)=1-dist/distmax;
            end
            weight = weight.^(param(1));
            G=tensor(rand_tensor.*weight*range);
            X=tensor(ttensor(G,K));
        else
            error('Unknown format.')
        end
    end
end