function results_table = cor_dim_batch(folderpath, m_max, tau)
% COR_DIM_BATCH Iterates through all .mat files in a folder, computes the
% correlation dimension using cor_dim, and classifies each signal.
%
% Inputs:
%   folderpath - String specifying the directory containing .mat files
%   m_max      - Maximum embedding dimension to test (recommend >= 7)
%   tau        - Time delay for phase space reconstruction (integer)
%
% Outputs:
%   results_table - A table containing the filename and its classification

% Find all .mat files in the specified folder
file_pattern = fullfile(folderpath, '*.mat');
mat_files = dir(file_pattern);
num_files = length(mat_files);

if num_files == 0
    error('No .mat files found in the specified directory: %s', folderpath);
end

% Initialize cell arrays to store results
file_names = cell(num_files, 1);
classifications = cell(num_files, 1);

fprintf('Batch processing %d files in: %s\n', num_files, folderpath);
fprintf('====================================================\n');

% Loop through each file
for k = 1:num_files
    current_file = mat_files(k).name;
    full_path = fullfile(folderpath, current_file);

    file_names{k} = current_file;
    fprintf('Processing file %d/%d: %s... ', k, num_files, current_file);

    try
        % Load the file
        data = load(full_path);

        % Extract the signal
        % Assumes the .mat file contains exactly one variable with the signal,
        % or that the primary variable is named 'x' or 'signal'
        vars = fieldnames(data);
        if isfield(data, 'x')
            x = data.x;
        elseif isfield(data, 'signal')
            x = data.signal;
        else
            % Default to the first variable found in the file
            x = data.(vars{1});
        end

        % Ensure it's a 1D numeric array before passing to cor_dim
        if isnumeric(x) && isvector(x)
            % Suppress figures during batch processing to avoid screen clutter
            % If you want to keep the figures, comment out the next line
            set(0, 'DefaultFigureVisible', 'off');

            % Call the provided cor_dim function
            [~, class_result] = cor_dim(x, m_max, tau);

            % Restore figure visibility
            set(0, 'DefaultFigureVisible', 'on');

            classifications{k} = class_result;
            fprintf('Done. (%s)\n', class_result);
        else
            classifications{k} = 'Error: Data is not a 1D numeric vector';
            fprintf('Skipped (Invalid data format).\n');
        end

    catch ME
        % Handle errors (e.g., time series too short) gracefully
        classifications{k} = sprintf('Error: %s', ME.message);
        fprintf('Failed (%s).\n', ME.message);

        % Ensure figure visibility is restored if it failed mid-execution
        set(0, 'DefaultFigureVisible', 'on');
    end
end

fprintf('====================================================\n');
fprintf('Batch processing complete.\n');

% Compile results into a MATLAB Table
results_table = table(file_names, classifications, ...
    'VariableNames', {'FileName', 'Classification'});

% Optional: Display the final table in the command window
disp(results_table);
end