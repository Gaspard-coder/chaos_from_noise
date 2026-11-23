% Define the directory containing the signals
folder_path = '/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/deter_test_sweep';

% Get a list of all .mat files in the directory
files = dir(fullfile(folder_path, '*.mat'));

% Preallocate a structure to hold the results
results = struct('FileName', {}, 'SignalType', {}, 'Fs', {}, 'LyapunovExponent', {});

fprintf('Starting Lyapunov Exponent computation...\n');
fprintf('--------------------------------------------------\n');

for i = 1:length(files)
    filename = files(i).name;
    filepath = fullfile(folder_path, filename);

    % Load the variable 'x' from the .mat file
    data = load(filepath, 'x');
    x = data.x;

    % N = 10 000 points[cite: 1]
    N = length(x);

    % Parse the signal type from the filename (e.g., DET_lorenz_x_r01.mat -> lorenz_x)
    % Find the position of the first underscore and the '_r' realization tag
    first_uscore = find(filename == '_', 1);
    r_tag = strfind(filename, '_r0');

    if isempty(r_tag)
        r_tag = strfind(filename, '_r1'); % In case of _r10, etc.
    end

    % Extract the core type string
    signal_type = filename(first_uscore+1 : r_tag(end)-1);

    % Assign the correct sampling frequency (Fs) based on the image rules
    switch signal_type
        case 'lorenz_x'
            fs = 50; %[cite: 1]
        case 'rossler_x'
            fs = 10; %[cite: 1]
        case {'linear', 'exponential'}
            fs = N - 1; %[cite: 1]
        case {'quadratic', 'polynomial'}
            fs = (N - 1) / 2; %[cite: 1]
        case {'sinusoid', 'two_tone', 'sawtooth', 'chirp', ...
                'logistic_map', 'henon_map', 'white_gaussian', ...
                'white_uniform', 'ar1_colored', 'smoothed_noise'}
            fs = 1; %[cite: 1]
        otherwise
            warning('Unknown signal type for %s, defaulting Fs to 1.', filename);
            fs = 1;
    end

    % Compute the Lyapunov Exponent
    try
        % The lyapunovExponent function is part of the Predictive Maintenance Toolbox
        lyap_exp = lyapunovExponent(x, fs);
    catch ME
        warning('Failed to compute LE for %s: %s', filename, ME.message);
        lyap_exp = NaN;
    end

    % Save the results
    results(i).FileName = filename;
    results(i).SignalType = signal_type;
    results(i).Fs = fs;
    results(i).LyapunovExponent = lyap_exp;

    fprintf('File: %-25s | Fs: %-6g | LE: %f\n', filename, fs, lyap_exp);
end

% Convert the results structure into a table for easy viewing and saving
resultsTable = struct2table(results);

% Optional: Save the results to a CSV file
% writetable(resultsTable, fullfile(folder_path, 'lyapunov_results.csv'));

fprintf('--------------------------------------------------\n');
fprintf('Computation finished.\n');