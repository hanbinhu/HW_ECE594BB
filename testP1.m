clear; clc; close all;
seed = 10;

dim = [3,4,2];
rank = 4;
lambda_range = 10;

verbose = true;

% Parameters for CP decomposition
rank_cp = 4;
maxiters_cp = 100;
abstol_cp = 1e-2;
difftol_cp = 1e-4;

% Set random seed
%rng(seed);

% Generate the low rank tensor
[X, lambda_X, K_X] = generate_low_rank_tensor(dim, rank, lambda_range);
[Y, lambda_Y, K_Y] = generate_low_rank_tensor(dim, rank, lambda_range);

% Get CP decomposition
[Xhat_CP, lambda_X_CP, K_X_CP, K_init_X_CP, converge_X_CP] = ...
    cp_decomp_als(X, rank_cp,...
        'maxiters', maxiters_cp, 'verbose', verbose,...
        'abstol', abstol_cp, 'difftol', difftol_cp);
[Yhat_CP, lambda_Y_CP, K_Y_CP, K_init_Y_CP, converge_Y_CP] = ...
    cp_decomp_als(Y, rank_cp,...
        'maxiters', maxiters_cp, 'verbose', verbose,...
        'abstol', abstol_cp, 'difftol', difftol_cp);    

% Inner production computation
ip_acc = innerprod(X,Y);
fprintf('Original inner product: %.4f\n', ip_acc);
ip_CP = innerprod(Xhat_CP,Yhat_CP);
fprintf(['Inner product by CP decomposition: %.4f, absolute error: %.4f'...
    ', relative error: %.4f\n'],...
    ip_CP, abs(ip_acc-ip_CP), abs(ip_acc-ip_CP)/abs(ip_acc));