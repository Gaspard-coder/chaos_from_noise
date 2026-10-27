function plot_multiple_HXC(m,t,data_folder)

% =========================================================================
% This script loads multiple .mat files from a specified directory. 
% Folder containing the .mat files (use './' for current folder)
% ex : './Database/ECG_signals/'
%  
% For each file, it extracts a 1D time series, computes the Normalized Shannon 
% Permutation Entropy (H) and MPR Complexity (Comp_JS) using the 
% 'HXC_1_opti' function, and plots them on the Complexity-Entropy plane.
%
% =========================================================================
% Input:
%     m: embedding dimension
%     t: embedding time delay
%     data_folder: the folder where are stored the .mat to be processed
% 
% Output:
%     C x H feature plan with the theoritical limits
% =========================================================================

% Compute Theoretical Boundary Curves for dimension m
fprintf('Calculating theoretical bounds for embedding dimension m = %d...\n', m);
[H_max, C_max, H_min, C_min] = compute_theoretical_bounds(m);

% Get a list of all .mat files
file_pattern = fullfile(data_folder, '*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files found in the specified directory: %s', data_folder);
end

H_values = zeros(num_files, 1);
C_values = zeros(num_files, 1);
file_names = cell(num_files, 1);

fprintf('Processing %d files...\n', num_files);

for k = 1:num_files
    % Extract filename and full path
    filename = file_list(k).name;
    file_names{k} = filename;
    file_path = fullfile(data_folder, filename);
    
    % Load the data from the .mat file using struct2cell to extract 
    % the first variable without knowing its name
    loaded_data = load(file_path);
    data_cells = struct2cell(loaded_data);
    
    if isempty(data_cells)
        warning('File %s is empty. Skipping.', filename);
        continue;
    end
    
    % Assuming the first variable in the .mat file is the time series
    time_series = data_cells{1}; 
    
    % Ensure data is a 1D column vector as required by HXC_1_opti
    time_series = time_series(:); 
    [pe, ~, Comp_JS] = HXC_1_opti(time_series, m, t);
    H_values(k) = pe;
    C_values(k) = Comp_JS;
    
    fprintf('File: %-20s | H: %.4f | C: %.4f\n', filename, pe, Comp_JS);
end

%Plotting
    figure('Name', 'Complexity-Entropy Plane with Theoretical Bounds', ...
           'Color', 'w', 'Position', [100, 100, 850, 650]);
    hold on; grid on;

    % Plot theoretical bounds
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

%% Helper Function: Compute Theoretical Curves
function [H_max, C_max, H_min, C_min] = compute_theoretical_bounds(m)
    dim = factorial(m);
    num_points = 500; % Resolution per segment
    
    % --- Upper bound curve (C_max) ---
    P_max = zeros((dim - 1) * num_points, dim);
    for k = 1:(dim - 1)
        p_vals = linspace(1/(k+1), 1/k, num_points)';
        
        % Block-vectorized assignment (faster)
        idx_start = (k - 1) * num_points + 1;
        idx_end   = k * num_points;
        
        P_max(idx_start:idx_end, 1:k) = repmat((1 - p_vals) / k, 1, k);
        P_max(idx_start:idx_end, k+1) = p_vals;
    end
    [H_max, C_max] = HXC_from_prob_vectorized(P_max);

    % Sort just in case, to guarantee a smooth line plot from left to right
    [H_max, sort_idx] = sort(H_max);
    C_max = C_max(sort_idx);


    % --- Lower bound curve (C_min) ---
    p_vals_min = linspace(1/dim, 1, num_points)';
    P_min = zeros(num_points, dim);
    P_min(:, 1) = p_vals_min;
    P_min(:, 2:dim) = repmat((1 - p_vals_min) / (dim - 1), 1, dim - 1);
    
    [H_min, C_min] = HXC_from_prob_vectorized(P_min);
end

%% Helper Function: Compute H and C from Probability Matrix (Vectorized)
function [H_perm, Comp_JS] = HXC_from_prob_vectorized(P)
    % P is a matrix of size [N_distributions x dim]
    dim = size(P, 2);
    entropy_max = log(dim);
    
    % Avoid log(0) by setting 0 values to 1 (since 1 * log(1) = 0)
    P_safe = P;
    P_safe(P_safe == 0) = 1;
    H_p = -sum(P .* log(P_safe), 2);
    H_perm = H_p / entropy_max;
    
    % Uniform distribution baseline
    Pe = ones(1, dim) / dim;
    H_pe = log(dim);
    
    % Mixture distribution
    M = 0.5 * (P + repmat(Pe, size(P, 1), 1));
    M_safe = M;
    M_safe(M_safe == 0) = 1;
    H_m = -sum(M .* log(M_safe), 2);
    
    % Jensen-Shannon Divergence
    JSD = H_m - 0.5 * H_p - 0.5 * H_pe;
    
    % Normalization constant
    c1 = (0.5 + ((1 - 0.5)/dim)) * log(0.5 + ((1 - 0.5)/dim));
    c2 = ((1 - 0.5)/dim) * log((1 - 0.5)/dim) * (dim - 1);
    c3 = 0.5 * log(dim);
    Q0 = -1 / (c1 + c2 + c3);
    
    Comp_JS = Q0 * JSD .* H_perm;
end