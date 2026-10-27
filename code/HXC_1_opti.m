function [pe, JSD, Comp_JS]=HXC_1_opti(data,m,t)
% =========================
% This matlab function is designed to quickly compute parameters of the 
% Complexity Entropy Causality plane. It uses regular Shannon permutation
% entropy and MPR complexity definition
% 
% It is inspired and modified from the work of Dr. Walter Legnani
% =========================
% Input:
%     data: one dimensional vector
%     m: embedding dimension to compute permtuation entropy (pe)
%     t: embedding time delay to compute pe as well
% 
% Output:
%     pe: normalized Shannon entropy H in log in order to fit
%     in [0;1] interval
%     JSD: Jansen Shannon Divergence to compute disequilibrium
%     Comp_JS: MPR complexity 
% ========================

% Ensure data is a column vector
data = data(:);

ly = length(data);
nV = ly - t*(m-1); 
offsets = (0:t:t*(m-1))'; 
starts  = 1:nV;           
idx = offsets + starts;   
Y = data(idx); 

% Get ordinal patterns for each vector
[~, IV] = sort(Y, 1);
IV = IV'; % nV x m

% Map extracted patterns to all possible m! permutations
all_patterns = perms(1:m); 
num_patterns = factorial(m);
[~, pattern_idx] = ismember(IV, all_patterns, 'rows');

c = accumarray(pattern_idx, 1, [num_patterns, 1]); 
p = c / sum(c); % Probabilistic distribution of the signal

% Compute Normalized Permutation Entropy (pe)
entropy_max = log(num_patterns);
p_pos = p(p > 0);
pe = -sum(p_pos .* log(p_pos)) / entropy_max;

% Compute Jensen-Shannon Divergence (JSD)
u = 1 / num_patterns; % Uniform distribution baseline

% H((P + U) / 2)
p_mix = 0.5 * p + 0.5 * u;
H_mix = -sum(p_mix .* log(p_mix));
% H(P)
H_p = -sum(p_pos .* log(p_pos));
% H(U)
H_u = log(num_patterns);

% JSD
JSD = H_mix - 0.5 * H_p - 0.5 * H_u;

% Compute MPR Complexity (Comp_JS)
% Normalization constant Q_0
c1 = (0.5 + ((1 - 0.5) / num_patterns)) * log(0.5 + ((1 - 0.5) / num_patterns));
c2 = ((1 - 0.5) / num_patterns) * log((1 - 0.5) / num_patterns) * (num_patterns - 1);
c3 = 0.5 * log(num_patterns);
Q_0 = -1 / (c1 + c2 + c3);

% MPR Complexity with normalized PE
Comp_JS = Q_0 * JSD * pe;

end
