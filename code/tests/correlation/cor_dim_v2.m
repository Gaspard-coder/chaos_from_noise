function [D2_vals, classification] = cor_dim_v2(x, m_max, tau)
% COR_DIM Computes the correlation dimension across embedding dimensions
% m = 2:m_max and classifies the signal as Chaotic, Deterministic, or Random.
%
% Inputs:
%   x     - 1D array containing the time series data
%   m_max - Maximum embedding dimension to test (recommend 9 or 10)
%   tau   - Time delay (integer) or 'auto' to estimate per signal via ACF

% Validate and prepare input
x = double(x(:));
x = x(~isnan(x));
N = length(x);

if m_max < 6
    error('m_max should be at least 6 (recommend 9-10) to reliably detect saturation.');
end

% Standardize signal
x = (x - mean(x)) / std(x);

% Adaptive tau estimation (first zero-crossing of autocorrelation)
if ischar(tau) || isstring(tau) || isempty(tau)
    acf = xcorr(x, 'coeff');
    acf = acf(N:end); % Keep non-negative lags
    zero_cross = find(acf <= 0, 1, 'first');
    if isempty(zero_cross)
        tau = 1;
    else
        tau = max(1, zero_cross - 1);
    end
    % Cap tau so at least half the time series remains usable at m_max
    max_allowed_tau = max(1, floor((N - 50) / (2 * (m_max - 1))));
    tau = min(tau, max_allowed_tau);
end

if N - (m_max - 1)*tau < 20
    error('Time series is too short for the specified m_max and tau.');
end

% Theiler window to avoid temporal correlation bias in slow IMFs
theiler_W = max(1, tau);

m_vals = 2:m_max;
num_m = length(m_vals);
D2_vals = nan(1, num_m);

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

    % 2. Pairwise Euclidean Distances with Theiler Window
    D = pdist(X, 'euclidean');
    [row, col] = find(tril(true(N_m), -1));
    D = D((row - col) > theiler_W);
    D = D(D > 0);

    % 3. Correlation Integral C(r)
    r = logspace(log10(min(D)), log10(max(D)), 60);
    total_pairs = length(D);
    Cr = arrayfun(@(rad) sum(D <= rad) / total_pairs, r);

    valid = Cr > 0;
    log_r = log(r(valid));
    log_Cr = log(Cr(valid));

    % 4. Estimate D2 in the linear scaling region
    scale_idx = find(Cr(valid) >= 0.005 & Cr(valid) <= 0.25);
    if length(scale_idx) < 5
        i1 = max(1, floor(length(log_r) * 0.3));
        i2 = min(length(log_r), floor(length(log_r) * 0.7));
        scale_idx = i1:i2;
    end

    if length(scale_idx) >= 2
        p = polyfit(log_r(scale_idx), log_Cr(scale_idx), 1);
        D2_vals(idx) = p(1);
    end

    plot(log_r, log_Cr, 'Color', colors(idx, :), 'LineWidth', 1.3, ...
        'DisplayName', sprintf('m = %d (D_2 = %.2f)', m, D2_vals(idx)));
end

xlabel('ln(r)');
ylabel('ln(C(r))');
title(sprintf('Correlation Integral Scaling Curves (\\tau = %d)', tau));
legend('Location', 'southeast', 'FontSize', 8);
grid on;
hold off;

% 5. Classify Signal Behavior from D2 Evolution
tail_m = m_vals(end-2:end);
tail_D2 = D2_vals(end-2:end);

if any(isnan(tail_D2))
    classification = 'Inconclusive (insufficient scaling data)';
    is_saturated = false;
else
    slope_fit = polyfit(tail_m, tail_D2, 1);
    tail_slope = slope_fit(1);
    D2_sat = D2_vals(end); % Use final converged value instead of tail mean

    % Tighter, physically grounded thresholds
    %saturation_slope_thresh = 0.05; % True plateau requires near-zero slope
    %integer_tol = 0.10;             % Tolerance around integer manifolds (2, 3, ...)

    %updated coefficients with cor_dim_sweep:
    saturation_slope_thresh = 0.6133;
    integer_tol = 0.0089;

    if tail_slope > saturation_slope_thresh
        is_saturated = false;
        classification = 'Random / Stochastic Noise';
        reason = sprintf('D_2 keeps growing with m (tail slope = %.3f > %.2f) without saturating.', ...
            tail_slope, saturation_slope_thresh);
    else
        is_saturated = true;
        % Continuous 1D periodic/trend curves saturate <= 1.0 (down to ~0.75 due to edge effects)
        if D2_sat <= (1 + integer_tol)
            classification = 'Deterministic Regular (1D Periodic / Trend)';
            reason = sprintf('D_2 saturates at %.2f (consistent with a 1D regular curve).', D2_sat);
        elseif abs(D2_sat - round(D2_sat)) <= integer_tol
            classification = 'Deterministic Regular (Quasi-periodic)';
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
fprintf('\n--- Correlation Dimension Analysis (tau = %d) ---\n', tau);
for idx = 1:num_m
    fprintf('m = %2d  -->  D2 = %.4f\n', m_vals(idx), D2_vals(idx));
end
fprintf('-------------------------------------------------\n');
if exist('reason', 'var')
    fprintf('Details: %s\n', reason);
end
end