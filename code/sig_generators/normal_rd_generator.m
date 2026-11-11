function [x] = normal_rd_generator()

% Normally distributed random number with zero mean and unit variance

N=10000;
sigma_eps = 1;
x = sigma_eps* randn(N,sigma_eps);

save('database/generated_sig/rand_sig.mat',"x");
fprintf('Normally distributed rand sig saved in rand_sig.mat\n');
end