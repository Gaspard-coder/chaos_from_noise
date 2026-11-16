function results = deter_test_batch(folder_path,m,tau,eps)
% DETER_TEST_BATCH Processes all .mat files in a folder to classify
% their 1D signals as deterministic or random/noisy using RQA.

% Find all .mat files in the specified directory
file_pattern = fullfile(folder_path, '*.mat');
mat_files = dir(file_pattern);
num_files = length(mat_files);

if num_files == 0
    error('No .mat files found in the directory: %s', folder_path);
end

% Preallocate data structures for the results
file_names = cell(num_files, 1);
det_values = zeros(num_files, 1);
classifications = cell(num_files, 1);

fprintf('Processing %d files...\n', num_files);

for k = 1:num_files
    base_file_name = mat_files(k).name;
    full_file_name = fullfile(folder_path, base_file_name);

    % Load the .mat file. This returns a struct with variables as fields.
    data = load(full_file_name);
    vars = fieldnames(data);

    % Extract the first variable in the file to use as the signal
    x = data.(vars{1});

    % Validate that the extracted data is a 1D vector
    if ~isvector(x) || isnumeric(x) == false
        warning('Variable "%s" in "%s" is not a valid 1D numeric array. Skipping.', vars{1}, base_file_name);
        file_names{k} = base_file_name;
        det_values(k) = NaN;
        classifications{k} = 'Invalid Data';
        continue;
    end

    % Ensure x is a column vector for consistency
    x = x(:);

    % Call the original determinism test function
    % NOTE: Ensure deter_test_2.m is in your MATLAB path.
    DET = deter_test_3(x,m,tau,eps);

    % Store the results
    file_names{k} = base_file_name;
    det_values(k) = DET;

    % Apply the classification threshold (40%)
    if DET > 40
        classifications{k} = 'Deterministic';
    else
        classifications{k} = 'Random/Noisy';
    end
end

% Compile results into a clean MATLAB Table
results = table(file_names, det_values, classifications, ...
    'VariableNames', {'FileName', 'DET_Percent', 'Classification'});

% Display the final table in the command window
disp(results);
end