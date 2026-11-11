function [x] = garch_process()

% Generate chaotic tent map .mat of 10000 points
N = 10000;
x = zeros(N,1);
h = zeros(N,1);

% White noise
sigma_eps = 0.01;
eps = sigma_eps * randn(N,1);

x(1) = 0;
h(1) = 0.2;

for i = 2:N
    h(i) = 0.01 + 0.1*x(i-1)^2 + 0.85*h(i-1);
    x(i) = eps(i)*sqrt(h(i));
end

% Saving
save('database/generated_sig/garch_process.mat','x');
fprintf('Garch process series saved in garch_process.mat\n');
end