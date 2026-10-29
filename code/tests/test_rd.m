function results = test_rd(folder, varargin)
% TEST_RD Runs the runstest function on time series data in files starting with 'IMF'.
%
% Syntax:
%   results = test_rd(folder)
%   results = test_rd(folder, v)
%   results = test_rd(folder, v, 'Alpha', 0.01)
%   results = test_rd(folder, 'Alpha', 0.01, 'Method', 'exact')
%
% Inputs:
%   folder   - Target directory path containing .mat files (string or char).
%   varargin - (Optional) Arguments passed directly to runstest:
%              v, 'Alpha', 'Method', 'Tail'.
%              If omitted, runstest uses its default parameters:
%              v = median(x), Alpha = 0.05, Method = 'approximate', Tail = 'both'.
%
% Output:
%   results  - MATLAB table containing Filename, Hypothesis Decision (H),
%              p-value (P), and Number of Runs (N_runs) for each file.

% Find all .mat files starting with 'IMF' in the specified folder
file_pattern = fullfile(folder, 'IMF*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files starting with "IMF" were found in: %s', folder);
end

% Preallocate result arrays
file_names = cell(num_files, 1);
h_vals     = zeros(num_files, 1);
p_vals     = zeros(num_files, 1);
n_runs     = zeros(num_files, 1);

fprintf('Running randomness test on %d files in "%s"...\n\n', num_files, folder);

% Loop through each IMF file
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

    % Execute runstest forwarding all optional arguments
    [h, p, stats] = runstest(time_series, varargin{:});

    h_vals(k) = h;
    p_vals(k) = p;
    n_runs(k) = stats.nruns;

    % Output results to Command Window
    if h == 1
        decision = 'REJECT Null (Non-Random)';
    else
        decision = 'FAIL TO REJECT Null (Random)';
    end

    fprintf('File: %-15s | h: %d | p-value: %.4f | Runs: %d | Decision: %s\n', ...
        filename, h, p, stats.nruns, decision);
end

% Pack output into a structured table
results = table(file_names, h_vals, p_vals, n_runs, ...
    'VariableNames', {'Filename', 'H_Decision', 'P_Value', 'RunsCount'});
end