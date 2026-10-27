function plot_multiple_HXC_cached(m, t, data_folder)
% =========================================================================
% Plots the Complexity-Entropy plane for time series in data_folder.
% Uses pre-computed or cached theoretical bounds for dimension m.
% =========================================================================

%Load or compute theoretical bounds
[H_max, C_max, H_min, C_min] = get_theoretical_bounds(m);

%Find .mat files starting with IMF
file_pattern = fullfile(data_folder, 'IMF*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files starting with "IMF" found in directory: %s', data_folder);
end

H_values = zeros(num_files, 1);
C_values = zeros(num_files, 1);
file_names = cell(num_files, 1);

fprintf('Processing %d files...\n', num_files);
for k = 1:num_files
    filename = file_list(k).name;
    file_names{k} = filename;
    file_path = fullfile(data_folder, filename);

    loaded_data = load(file_path);
    data_cells = struct2cell(loaded_data);

    if isempty(data_cells)
        warning('File %s is empty. Skipping.', filename);
        continue;
    end

    time_series = data_cells{1}(:);
    [pe, ~, Comp_JS] = HXC_1_opti(time_series, m, t);

    H_values(k) = pe;
    C_values(k) = Comp_JS;

    fprintf('File: %-20s | H: %.4f | C: %.4f\n', filename, pe, Comp_JS);
end

% Plotting
figure('Name', 'Complexity-Entropy Plane with Theoretical Bounds', ...
    'Color', 'w', 'Position', [100, 100, 850, 650]);
hold on; grid on;

% Plot theoretical boundary curves
h_cmax = plot(H_max, C_max, 'k-', 'LineWidth', 2, 'DisplayName', 'C_{max}');
h_cmin = plot(H_min, C_min, 'r-', 'LineWidth', 2, 'DisplayName', 'C_{min}');

% Plot empirical data points
colors = lines(num_files);
data_handles = zeros(num_files, 1);
clean_names = cellfun(@(x) strrep(x, '.mat', ''), file_names, 'UniformOutput', false);

for k = 1:num_files
    data_handles(k) = scatter(H_values(k), C_values(k), 120, ...
        'Marker', 'x', ...
        'MarkerEdgeColor', colors(k,:), ...
        'LineWidth', 2, ...
        'DisplayName', clean_names{k});
end

% Formatting
xlabel('Normalized Permutation Entropy (H)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Jensen-Shannon Complexity (C)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Complexity-Entropy Plane (m = %d, t = %d)', m, t), ...
    'FontSize', 14, 'FontWeight', 'bold');

xlim([0 1]);
max_c = max([C_max(:); C_values(:)]);
ylim([0 max_c * 1.1]);
legend([h_cmax; h_cmin; data_handles], 'Interpreter', 'none', 'Location', 'northeastoutside');
hold off;
end

%% Helper Function: Cache / Compute Theoretical Bounds
function [H_max, C_max, H_min, C_min] = get_theoretical_bounds(m)
cache_file = sprintf('HXC_bounds_m%d.mat', m);

% Load from disk if cached
if exist(cache_file, 'file')
    fprintf('Loading theoretical bounds from cache file (%s)...\n', cache_file);
    cached = load(cache_file);
    H_max = cached.H_max;
    C_max = cached.C_max;
    H_min = cached.H_min;
    C_min = cached.C_min;
    return;
end

% Otherwise, calculate with memory-optimized resolution
fprintf('Calculating theoretical bounds for m = %d (one-time operation)...\n', m);
dim = factorial(m);

% Adaptive resolution: keeps total points around ~5,000 to ~10,000 max
num_points = max(5, round(5000 / dim));

% --- Upper bound curve (C_max) ---
num_segments = dim - 1;
P_max = zeros(num_segments * num_points, dim);

for k = 1:num_segments
    p_vals = linspace(1/(k+1), 1/k, num_points)';
    idx_start = (k - 1) * num_points + 1;
    idx_end   = k * num_points;

    % Fixed using repmat for multi-column block assignment
    P_max(idx_start:idx_end, 1:k) = repmat((1 - p_vals) / k, 1, k);
    P_max(idx_start:idx_end, k+1) = p_vals;
end

[H_max, C_max] = HXC_from_prob_vectorized(P_max);
[H_max, sort_idx] = sort(H_max);
C_max = C_max(sort_idx);

% --- Lower bound curve (C_min) ---
p_vals_min = linspace(1/dim, 1, num_points * 10)';
P_min = zeros(length(p_vals_min), dim);
P_min(:, 1) = p_vals_min;

% Fixed using repmat for multi-column block assignment
P_min(:, 2:dim) = repmat((1 - p_vals_min) / (dim - 1), 1, dim - 1);

[H_min, C_min] = HXC_from_prob_vectorized(P_min);

% Save to disk for future runs
save(cache_file, 'H_max', 'C_max', 'H_min', 'C_min');
fprintf('Theoretical bounds saved to %s.\n', cache_file);
end

%% Helper Function: Memory-Efficient H and C Computation
function [H_perm, Comp_JS] = HXC_from_prob_vectorized(P)
dim = size(P, 2);
entropy_max = log(dim);

% In-place computation of P * log(P) to avoid memory duplication
P_logP = zeros(size(P));
pos_mask = P > 0;
P_logP(pos_mask) = P(pos_mask) .* log(P(pos_mask));

H_p = -sum(P_logP, 2);
H_perm = H_p / entropy_max;

% Baseline uniform entropy component
Pe_val = 1 / dim;
H_pe = log(dim);

% Mixture distribution M = 0.5 * (P + 1/dim)
M = 0.5 * (P + Pe_val);
H_m = -sum(M .* log(M), 2);

% Jensen-Shannon Divergence
JSD = H_m - 0.5 * H_p - 0.5 * H_pe;

% Normalization constant Q0
c1 = (0.5 + ((1 - 0.5)/dim)) * log(0.5 + ((1 - 0.5)/dim));
c2 = ((1 - 0.5)/dim) * log((1 - 0.5)/dim) * (dim - 1);
c3 = 0.5 * log(dim);
Q0 = -1 / (c1 + c2 + c3);

Comp_JS = Q0 * JSD .* H_perm;
end