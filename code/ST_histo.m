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
%   results  - Table summarizing RandomCount, RandomFiles and TotalIMFs per subfolder.

items = dir(folder);
subfolders = items([items.isdir] & ~startsWith({items.name}, '.'));
num_subfolders = length(subfolders);

if num_subfolders == 0
    error('No subfolders in: %s',folder);
end

% Sort subfolders naturally (e.g., EMD_1, EMD_2, ..., EMD_10)
subfolder_names_raw = {subfolders.name}';
nums = zeros(num_subfolders, 1);
for k = 1:num_subfolders
    val = sscanf(subfolder_names_raw{k}, '%*[^0-9]%d');
    if ~isempty(val), nums(k) = val(1); else, nums(k) = k; end
end
[~, sortIdx] = sort(nums);
subfolders = subfolders(sortIdx);

subfolder_names   = cell(num_subfolders, 1);
random_counts = zeros(num_subfolders, 1);
total_imfs_counts = zeros(num_subfolders, 1);
random_files = strings(num_subfolders,1);

for sf = 1:num_subfolders
    subfolder_name = subfolders(sf).name;
    subfolder_path = fullfile(folder, subfolder_name);
    subfolder_names{sf} = subfolder_name;

    try
        % Execute test_rd forwarding varargin options
        res = test_rd(subfolder_path, varargin{:});

        % Count non-random IMFs (H = 1) and total IMFs
        is_random = (res.H_Decision == 0);
        rand_list = res.Filename(is_random);

        random_counts(sf) = sum(is_random);
        total_imfs_counts(sf) = height(res);

        % Join file names into a single string for clean display
        if isempty(rand_list)
            random_files(sf) = "None";
        else
            if iscell(rand_list)
                random_files(sf) = string(strjoin(rand_list, ', '));
            else
                random_files(sf) = string(strjoin(cellstr(rand_list), ', '));
            end
        end

    catch ME
        warning('Could not process subfolder "%s": %s', subfolder_name, ME.message);
        random_counts(sf) = NaN;
        total_imfs_counts(sf) = NaN;
        random_files(sf) = "Error";
    end
end

results = table(subfolder_names, random_counts, total_imfs_counts, random_files, ...
    'VariableNames', {'Subfolder', 'RandomCount', 'TotalIMFs', 'RandomFiles'});

plot_counts = random_counts;
plot_counts(isnan(plot_counts)) = 0;

cat_names = categorical(subfolder_names, subfolder_names);

figure('Name', 'Random IMF Components per Subfolder', 'Color', 'w');
b = bar(cat_names, plot_counts, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'k');

% Add explicit value labels above every bar
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(plot_counts);
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

xlabel('Subfolder', 'FontWeight', 'bold');
ylabel('Number of Random Components (H = 0)', 'FontWeight', 'bold');
title(sprintf('Number of Random IMFs per Subfolder in "%s"', folder), ...
    'Interpreter', 'none');

% Force y-axis range from 1 to 10 with integer tick marks
ylim([0, 10]);
yticks(1:10);

xtickangle(45); % Rotate x-axis text for better readability
grid on;
end
