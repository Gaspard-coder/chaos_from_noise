function [x] = tent_map()

% Generate chaotic tent map .mat of 10000 points
N = 10000;
x = zeros(N,1);

% White noise
sigma_eps = 0.01;
rng(1);
eps = sigma_eps * randn(N,1);

x(1) = 0.7;

for i = 2:N
    if x(i-1) < 0.5
        val = 2*x(i-1) + eps(i);
    else
        val = 2*(1-x(i-1)) + eps(i);
    end
    % prevent negativ values due to noise leading to -inf
    % small clipping at borders
    x(i) = max(0,min(1,val));
end

% Saving
save('database/generated_sig/tent_map.mat','x');
fprintf('Tent map series saved in tent_map.mat \n');
end