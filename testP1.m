clear; clc; close all;
% General Parameters
fixseed = true;
seed = 2;
verbose = true;

% Parameters for Problem Generation
dim = [20,50,30];
range = 10;
rank_k = 8;
rank_t = [8,9,10];
%form = 'Kruskal';
form = 'Tucker';

% Parameters for CP decomposition
rank_cp = 30;
maxiters_cp = 100;
abstol_cp = 1e-2;
difftol_cp = 1e-4;

% Parameter for Tucker decomposition
rank_tucker = [7,9,6];
maxiters_tucker = 20;
abstol_tucker = 1e-2;
difftol_tucker = 1e-4;

% Set random seed
if fixseed
    rng(seed);
else
    seed = rng;
end

% Generate the low rank tensor
if strcmp(form, 'Kruskal')
    [X, l_X, K_X] = generate_low_rank_tensor(dim, rank_k, range, form);
    [Y, l_Y, K_Y] = generate_low_rank_tensor(dim, rank_k, range, form);
end
if strcmp(form, 'Tucker')
    [X, G_X, K_X] = generate_low_rank_tensor(dim, rank_t, range, form);
    [Y, G_Y, K_Y] = generate_low_rank_tensor(dim, rank_t, range, form);
end

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