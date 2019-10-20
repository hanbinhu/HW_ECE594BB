clear; clc; close all;
seed = 10;

dim = [10,20,15];
rank = 8;
lambda_range = 10;

verbose = false;

% Parameters for CP decomposition
rank_cp = 10;
maxiters_cp = 100;
abstol_cp = 1e-2;
difftol_cp = 1e-4;

% Parameter for Tucker decomposition
%rank_tucker = [7,14,10];
rank_tucker = [3,6,5];
maxiters_tucker = 200;
abstol_tucker = 1e-2;
difftol_tucker = 1e-4;

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
    
% Get Tucker decomposition via HOSVD
[Xhat_HOSVD, G_X_HOSVD, K_X_HOSVD] = ...
    tucker_hosvd(X, rank_tucker, 'verbose', verbose);
[Yhat_HOSVD, G_Y_HOSVD, K_Y_HOSVD] = ...
    tucker_hosvd(Y, rank_tucker, 'verbose', verbose);

% Get Tucker decomposition via HOOI
[Xhat_HOOI, G_X_HOOI, K_X_HOOI, K_init_X_HOOI, converge_X_HOOI] = ...
    tucker_hooi(X, rank_tucker, 'verbose', true,...
        'abstol', abstol_tucker, 'difftol', difftol_tucker,...
        'maxiters', maxiters_tucker);
[Yhat_HOOI, G_Y_HOOI, K_Y_HOOI, K_init_Y_HOOI, converge_Y_HOOI] = ...
    tucker_hooi(Y, rank_tucker, 'verbose', true,...
        'abstol', abstol_tucker, 'difftol', difftol_tucker,...
        'maxiters', maxiters_tucker);

% Get Results from the tensor toolbox
[X_c,~,~]=cp_als(X, rank_cp, 'maxiters', maxiters_cp, 'tol', difftol_cp);
X_c=tensor(X_c);
[Y_c,~,~]=cp_als(Y, rank_cp, 'maxiters', maxiters_cp, 'tol', difftol_cp);
Y_c=tensor(Y_c);
X_s=tensor(hosvd(X, norm(X), 'ranks', rank_tucker));
Y_s=tensor(hosvd(Y, norm(Y), 'ranks', rank_tucker));
[X_o,~]=tucker_als(X, rank_tucker, 'maxiters', maxiters_tucker, 'tol', difftol_tucker);
X_o=tensor(X_o);
[Y_o,~]=tucker_als(Y, rank_tucker, 'maxiters', maxiters_tucker, 'tol', difftol_tucker);
Y_o=tensor(Y_o);

% Inner production computation
ip_acc = innerprod(X,Y);
fprintf('Original inner product: %.4f\n', ip_acc);
ip_CP = innerprod(Xhat_CP,Yhat_CP);
fprintf(['Inner product by CP decomposition: %.4f, absolute error: %.4f'...
    ', relative error: %.4f\n'],...
    ip_CP, abs(ip_acc-ip_CP), abs(ip_acc-ip_CP)/abs(ip_acc));
ip_HOSVD = innerprod(Xhat_HOSVD,Yhat_HOSVD);
fprintf(['Inner product by Tucker decomposition (HOSVD): %.4f, '...
    ' absolute error: %.4f, relative error: %.4f\n'],...
    ip_HOSVD, abs(ip_acc-ip_HOSVD), abs(ip_acc-ip_HOSVD)/abs(ip_acc));
ip_HOOI = innerprod(Xhat_HOOI,Yhat_HOOI);
fprintf(['Inner product by Tucker decomposition (HOOI): %.4f, '...
    ' absolute error: %.4f, relative error: %.4f\n'],...
    ip_HOOI, abs(ip_acc-ip_HOOI), abs(ip_acc-ip_HOOI)/abs(ip_acc));

ip_c = innerprod(X_c,Y_c);
fprintf(['Inner product by CP decomposition (standard): %.4f, '...
    ' absolute error: %.4f, relative error: %.4f\n'],...
    ip_c, abs(ip_acc-ip_c), abs(ip_acc-ip_c)/abs(ip_acc));
ip_s = innerprod(X_s,Y_s);
fprintf(['Inner product by Tucker decomposition (HOSVD, standard): %.4f, '...
    ' absolute error: %.4f, relative error: %.4f\n'],...
    ip_s, abs(ip_acc-ip_s), abs(ip_acc-ip_s)/abs(ip_acc));
ip_o = innerprod(X_o,Y_o);
fprintf(['Inner product by Tucker decomposition (HOOI, standard): %.4f, '...
    ' absolute error: %.4f, relative error: %.4f\n'],...
    ip_o, abs(ip_acc-ip_o), abs(ip_acc-ip_o)/abs(ip_acc));