function results = test_ch(folder, varargin)
% TEST_CH Runs the chaostest function on time series data in files starting with 'IMF'.
%
% Syntax:
%   results = test_ch(folder)
%   results = test_ch(folder, ActiveFN, maxOrders, alpha, flag)
%
% Inputs:
%   folder   - Target directory path containing .mat files (string or char).
%   varargin - (Optional) Passed directly to chaostest in order:
%              ActiveFN, maxOrders, alpha, flag.
%              If omitted, chaostest applies its default parameters:
%              ActiveFN = 'tanh', maxOrders = [5,6,5], alpha = 0.05, flag = 'VARY'.
%
% Output:
%   results  - MATLAB table containing Filename, H Decision, p-value,
%             Dominant Lyapunov Exponent (Lambda), and Best Orders [L, m, q].

% Find all .mat files starting with 'IMF'
file_pattern = fullfile(folder, 'IMF*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files starting with "IMF" found in directory: %s', folder);
end

% Preallocate output variables
file_names  = cell(num_files, 1);
H_vals      = zeros(num_files, 1);
p_vals      = zeros(num_files, 1);
Lambda_vals = zeros(num_files, 1);
Orders_cell = cell(num_files, 1);

fprintf('Running chaostest on %d files in "%s"...\n\n', num_files, folder);

% Process each file
for k = 1:num_files
    filename = file_list(k).name;
    file_names{k} = filename;
    file_path = fullfile(folder, filename);

    % Load data from .mat file
    loaded_struct = load(file_path);
    data_cells = struct2cell(loaded_struct);

    if isempty(data_cells)
        warning('File %s is empty. Skipping.', filename);
        continue;
    end

    % Extract time series vector
    time_series = data_cells{1}(:);

    % 3. Call chaostest with user parameters or default parameters
    [H, pValue, Lambda, Orders] = chaostest_opti(time_series, varargin{:});

    H_vals(k)      = H;
    p_vals(k)      = pValue;
    Lambda_vals(k) = Lambda;
    Orders_cell{k} = sprintf('[%d, %d, %d]', Orders(1), Orders(2), Orders(3));

    % Diagnostic interpretation
    if H == 0
        decision = 'Do NOT Reject Chaos (Chaotic)';
    else
        decision = 'Reject Chaos (Non-Chaotic)';
    end

    fprintf('File: %-15s | H: %d | p-value: %.4f | Lambda: %8.4f | Orders: %-9s | %s\n', ...
        filename, H, pValue, Lambda, Orders_cell{k}, decision);
end

% Pack output into a structured table
results = table(file_names, H_vals, p_vals, Lambda_vals, Orders_cell, ...
    'VariableNames', {'Filename', 'H_Decision', 'P_Value', 'Lambda_Exponent', 'Best_Orders'});
end