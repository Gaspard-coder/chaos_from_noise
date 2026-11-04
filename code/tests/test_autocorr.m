function results = test_autocorr(folder, lag_range)
% TEST_AUTOCORR Computes and plots the autocorrelation function for all files
% starting with 'IMF' in a specified subfolder across the given lag range.
%
% Syntax:
%   test_autocorr(folder, lag_range)
%   results = test_autocorr(folder, lag_range)
%
% Inputs:
%   folder    - Directory path containing 'IMF*.mat' files (string/char).
%   lag_range - Maximum lag value (scalar integer, e.g. 100 or 500).
%
% Output:
%   results   - MATLAB table containing Filename, 1/e Decorrelation Lag,
%               and First Zero-Crossing Lag for each IMF.

% Find all .mat files starting with 'IMF'
file_pattern = fullfile(folder, 'IMF*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files starting with "IMF" were found in: %s', folder);
end

% Sort files naturally (IMF_1, IMF_2, ..., IMF_10)
file_names_raw = {file_list.name}';
nums = zeros(num_files, 1);
for k = 1:num_files
    val = sscanf(file_names_raw{k}, '%*[^0-9]%d');
    if ~isempty(val), nums(k) = val(1); else, nums(k) = k; end
end
[~, sortIdx] = sort(nums);
file_list = file_list(sortIdx);

% Parse maximum lag scalar
if isscalar(lag_range)
    max_lag = lag_range;
else
    max_lag = max(abs(lag_range));
end

% Preallocate output table fields
file_names = cell(num_files, 1);
lag_1e     = zeros(num_files, 1);
lag_zero   = zeros(num_files, 1);

% Setup Figure
figure('Name', sprintf('Autocorrelation Analysis - %s', folder), 'Color', 'w');
hold on;
colors = lines(num_files);
legend_labels = cell(num_files, 1);

% Compute AutoCorrelation Function (ACF) and plot for each IMF file
for k = 1:num_files
    filename = file_list(k).name;
    file_names{k} = filename;
    file_path = fullfile(folder, filename);

    % Load signal data
    loaded_struct = load(file_path);
    data_cells = struct2cell(loaded_struct);
    if isempty(data_cells)
        warning('File %s is empty. Skipping.', filename);
        continue;
    end
    time_series = data_cells{1}(:);
    time_series = time_series - mean(time_series); % Zero-mean centering

    % Compute normalized autocorrelation
    [raw_acf, full_lags] = xcorr(time_series, max_lag, 'coeff');

    % Keep non-negative lags (0 to max_lag)
    zero_idx = find(full_lags == 0);
    acf  = raw_acf(zero_idx:end);
    lags = full_lags(zero_idx:end);

    % Extract characteristic decorrelation lags
    idx_e = find(acf <= 1/exp(1), 1);
    if ~isempty(idx_e), lag_1e(k) = lags(idx_e); else, lag_1e(k) = NaN; end

    idx_z = find(acf <= 0, 1);
    if ~isempty(idx_z), lag_zero(k) = lags(idx_z); else, lag_zero(k) = NaN; end

    % Format legend label
    clean_label = strrep(filename, '.mat', '');
    clean_label = strrep(clean_label, '_', ' ');
    legend_labels{k} = clean_label;

    % Plot ACF curve
    plot(lags, acf, 'LineWidth', 1.5, 'Color', colors(k, :));
end

% Plot Formatting and Threshold Lines
yline(1/exp(1), '--r', '1/e Decay (0.368)', ...
    'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
yline(0, '-k', 'LineWidth', 1);

xlabel('Lag (\tau)', 'FontWeight', 'bold');
ylabel('Autocorrelation R_{xx}(\tau)', 'FontWeight', 'bold');
title(sprintf('Autocorrelation Functions for IMFs in "%s"', folder), ...
    'Interpreter', 'none', 'FontWeight', 'bold');

legend(legend_labels, 'Location', 'northeastoutside', 'Interpreter', 'none');
grid on;
xlim([0, max_lag]);
ylim([-1.05, 1.05]);
hold off;

% 5. Return Summary Table
results = table(file_names, lag_1e, lag_zero, ...
    'VariableNames', {'Filename', 'DecorrelationLag_1e', 'FirstZeroCrossingLag'});
end