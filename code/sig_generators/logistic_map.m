function [x] = logistic_map()

% Generate chaotic logistic map .mat of 10000 points
N = 10000;
x = zeros(N,1);

% White noise
sigma_eps = 0.01;
rng(1);
eps = sigma_eps * randn(N,1);

x(1) = 0.7;

for i = 2:N
    x(i) = 3.57*x(i-1)*(1-x(i-1))+ eps(i) ;
end

% Saving
save('database/generated_sig/logistic_map.mat','x');
fprintf('Logistic map series saved in logistic_map.mat\n');
end