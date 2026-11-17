function [x] = arma_process()

% Generate non chaotic arma process .mat of 10000 points
N = 10000;
x = zeros(N,1);

% White noise
sigma_eps = 0.01;
rng(1);
eps = sigma_eps * randn(N,1);

eps(1) = 0;
x(1) = 0.1;

for i = 2:N
    x(i) = 0.1 + 0.2*x(i-1) + eps(i)+0.15*eps(i-1);
end

% Saving
save('database/generated_sig/arma_process.mat','x');
fprintf('Arma process series saved in arma_process.mat\n');
end