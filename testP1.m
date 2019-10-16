clear; clc; close all;
seed = 10;
dim = [3,4,2];
rank = 4;
rank_cp = 4;
lambda_range = 10;

% Set random seed
rng(seed);

% Generate the low rank tensor
[X, lambda_X, K_X] = generate_low_rank_tensor(dim, rank, lambda_range);
[Xhat_CP, lambda_X_CP, K_X_CP, K_init_X_CP] = cp_decomp_als(X, rank_cp);

[Y, lambda_Y, K_Y] = generate_low_rank_tensor(dim, rank, lambda_range);
[Yhat_CP, lambda_Y_CP, K_Y_CP, K_init_Y_CP] = cp_decomp_als(Y, rank_cp);

ip_acc = innerprod(X,Y);
ip_app = innerprod(Xhat_CP,Yhat_CP);

display(ip_acc);
display(ip_app);

