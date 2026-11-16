function [D2_vals, classification] = cor_dim(x, m_max, tau)
% CORR_DIM Computes the correlation dimension across embedding dimensions
% m = 2:m_max and classifies the signal as Chaotic, Deterministic, or Random.
%
% Inputs:
%   x     - 1D array containing the time series data
%   m_max - Maximum embedding dimension to test (recommend >= 7)
%   tau   - Time delay for phase space reconstruction (integer)
%
% Outputs:
%   D2_vals        - Vector of estimated D2 values for m = 2:m_max
%   classification - String concluding the nature of the signal

% Validate and prepare input
x = double(x(:));
x = x(~isnan(x)); % Remove any NaNs
N = length(x);

if m_max < 5
    error('m_max must be at least 5 to reliably detect saturation.');
end
if N - (m_max - 1)*tau < 10
    error('Time series is too short for the specified m_max and tau.');
end

% Standardize signal to keep distance scales consistent
x = (x - mean(x)) / std(x);

m_vals = 2:m_max;
num_m = length(m_vals);
D2_vals = nan(1, num_m);

% Prepare figure
figure('Position', [100, 100, 1100, 450]);

% Subplot 1: Log-Log Correlation Integrals
subplot(1, 2, 1);
hold on;
colors = lines(num_m);

for idx = 1:num_m
    m = m_vals(idx);

    % 1. Phase Space Reconstruction (Takens' Embedding)
    N_m = N - (m - 1)*tau;
    X = zeros(N_m, m);
    for j = 1:m
        X(:, j) = x(1 + (j - 1)*tau : N_m + (j - 1)*tau);
    end

    % 2. Pairwise Euclidean Distances
    D = pdist(X, 'euclidean');
    D = D(D > 0);

    % 3. Correlation Integral C(r)
    r = logspace(log10(min(D)), log10(max(D)), 60);
    total_pairs = length(D);
    Cr = arrayfun(@(rad) sum(D <= rad) / total_pairs, r);

    valid = Cr > 0;
    log_r = log(r(valid));
    log_Cr = log(Cr(valid));

    % 4. Estimate D2 in the linear scaling region
    % Focus on C(r) between 0.005 and 0.25 to avoid noise floor & boundary effects
    scale_idx = find(Cr(valid) >= 0.005 & Cr(valid) <= 0.25);

    % Fallback to middle 40% if series is too short for strict C(r) bounds
    if length(scale_idx) < 5
        i1 = max(1, floor(length(log_r) * 0.3));
        i2 = min(length(log_r), floor(length(log_r) * 0.7));
        scale_idx = i1:i2;
    end

    if length(scale_idx) >= 2
        p = polyfit(log_r(scale_idx), log_Cr(scale_idx), 1);
        D2_vals(idx) = p(1);
    end

    % Plot log-log curve for this m
    plot(log_r, log_Cr, 'Color', colors(idx, :), 'LineWidth', 1.3, ...
        'DisplayName', sprintf('m = %d (D_2 = %.2f)', m, D2_vals(idx)));
end

xlabel('ln(r)');
ylabel('ln(C(r))');
title('Correlation Integral Scaling Curves');
legend('Location', 'southeast', 'FontSize', 8);
grid on;
hold off;

% 5. Classify Signal Behavior from D2 Evolution
% Analyze the slope over the final 3 embedding dimensions
tail_m = m_vals(end-2:end);
tail_D2 = D2_vals(end-2:end);

if any(isnan(tail_D2))
    classification = 'Inconclusive (insufficient scaling data)';
    is_saturated = false;
else
    slope_fit = polyfit(tail_m, tail_D2, 1);
    tail_slope = slope_fit(1);
    D2_sat = mean(tail_D2);

    % Thresholds for classification
    saturation_slope_thresh = 0.35; % Below this, D2 is considered plateaued
    integer_tol = 0.12;             % Tolerance for integer vs. fractal dimension

    if tail_slope > saturation_slope_thresh
        is_saturated = false;
        classification = 'Random / Stochastic Noise';
        reason = sprintf('D_2 keeps growing with m (tail slope = %.2f > %.2f) without saturating.', ...
            tail_slope, saturation_slope_thresh);
    else
        is_saturated = true;
        dist_to_int = abs(D2_sat - round(D2_sat));
        if dist_to_int <= integer_tol
            classification = 'Deterministic Regular (Periodic / Quasi-periodic)';
            reason = sprintf('D_2 saturates around %.2f (close to integer %d).', ...
                D2_sat, round(D2_sat));
        else
            classification = 'Chaotic (Deterministic Chaos)';
            reason = sprintf('D_2 saturates at a non-integer fractal value (D_2 \\approx %.2f).', ...
                D2_sat);
        end
    end
end

% Subplot 2: Evolution of D2 vs m
subplot(1, 2, 2);
plot(m_vals, D2_vals, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'Estimated D_2');
hold on;
% Reference line for ideal white noise (D2 = m)
plot(m_vals, m_vals, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Random Noise Limit (D_2 = m)');

if is_saturated
    yline(D2_sat, 'r--', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Saturation D_2 \\approx %.2f', D2_sat));
end

xlabel('Embedding Dimension (m)');
ylabel('Correlation Dimension (D_2)');
title({'Evolution of D_2 vs. m', ['Conclusion: ', classification]});
legend('Location', 'northwest');
grid on;
hold off;

% Print summary to Command Window
fprintf('\n--- Correlation Dimension Analysis ---\n');
for idx = 1:num_m
    fprintf('m = %2d  -->  D2 = %.4f\n', m_vals(idx), D2_vals(idx));
end
fprintf('--------------------------------------\n');