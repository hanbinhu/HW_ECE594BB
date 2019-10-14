clear; clc; close all;
seed=3;
dim=[3,4,2];
rank=4;
lambda_range=10;

% Set random seed
rng(seed);

% Generate the low rank tensor
[X, lambda, K]=generate_low_rank_tensor(dim, rank, lambda_range);