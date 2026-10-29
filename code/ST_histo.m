function results = ST_histo(folder, varargin)
% ST_HISTO Runs test_rd across all subfolders and plots a histogram of the
% number of random components (H = 0) found in each subfolder.
%
% Syntax:
%   results = ST_HISTO(folder)
%   results = ST_HISTO(folder, 'Alpha', 0.01)
%   results = ST_HISTO(folder, varargin)
%
% Inputs:
%   folder   - Parent folder path containing EMD subfolders (string/char).
%   varargin - Optional arguments forwarded to test_rd (e.g., 'Alpha', 0.01).
%
% Output:
%   results  - Table summarizing RandomCount and TotalIMFs per subfolder.

items = dir(folder);
subfolders = items([items.isdir] & ~startsWith({items.name}, '.'));
num_subfolders = length(subfolders);

if num_subfolders == 0
    error('No subfolders in: %s',folder);
end

subfolder_names   = cell(num_subfolders, 1);
random_counts = zeros(num_subfolders, 1);
total_imfs_counts = zeros(num_subfolders, 1);

for sf = 1:num_subfolders
    subfolder_name = subfolders(sf).name;
    subfolder_path = fullfile(folder, subfolder_name);
    subfolder_names{sf} = subfolder_name;

    try
        % Execute test_rd forwarding varargin options
        res = test_rd(subfolder_path, varargin{:});

        % Count non-random IMFs (H = 1) and total IMFs
        random_counts(sf) = sum(res.H_Decision == 0);
        total_imfs_counts(sf) = height(res);

    catch ME
        warning('Could not process subfolder "%s": %s', subfolder_name, ME.message);
        random_counts(sf) = NaN;
        total_imfs_counts(sf) = NaN;
    end
end

results = table(subfolder_names, random_counts, total_imfs_counts, ...
    'VariableNames', {'Subfolder', 'RandomCount', 'TotalIMFs'});

valid_mask  = ~isnan(random_counts);
plot_names  = subfolder_names(valid_mask);
plot_counts = random_counts(valid_mask);

cat_names = categorical(plot_names, plot_names);

figure('Name', 'Random IMF Components per Subfolder', 'Color', 'w');
bar(cat_names, plot_counts, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'k');

xlabel('Subfolder', 'FontWeight', 'bold');
ylabel('Number of Random Components (H = 0)', 'FontWeight', 'bold');
title(sprintf('Number of Random IMFs per Subfolder in "%s"', folder), ...
    'Interpreter', 'none');

xtickangle(45); % Rotate x-axis text for better readability
grid on;
end
