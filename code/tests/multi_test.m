function multi_test(folder_path)
file_pattern = fullfile(folder_path, '*.mat');
file_list = dir(file_pattern);
num_files = length(file_list);

if num_files == 0
    error('No .mat files found in the directory: %s', folder_path);
end
fprintf("Running runstest\n");
test_rd(folder_path);
fprintf("Computing H and C\n");
plot_multiple_HXC_cached(6,1,folder_path);
fprintf("Running deter test\n");
deter_test_batch(folder_path,6,2,0.01);
fprintf("Running correlation dimension test\n");
cor_dim_batch_v2(folder_path,8,1);

fprintf("Computing the raw lyapunov exponent\n");
% raw lyapunov exponent
file_names = cell(num_files, 1);
lp_vals= zeros(num_files,1);

for k = 1:num_files
    filename = file_list(k).name;
    file_names{k} = filename;
    file_path = fullfile(folder_path, filename);

    % Load data from .mat file
    loaded_struct = load(file_path);
    data_cells = struct2cell(loaded_struct);

    if isempty(data_cells)
        warning('File %s is empty. Skipping.', filename);
        continue;
    end

    % Extract time series vector
    lp_vals(k) = lyapunov_exponent(file_path);
    fprintf('File: %-15s | lambda: %d\n',filename,lp_vals(k));
end