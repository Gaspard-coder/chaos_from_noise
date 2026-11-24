% Define the directory containing the signals
folder_path = '/home/gaspichou/Documents/git/chaos_from_noise/database/generated_sig/deter_test_sweep';

% Get a list of all .mat files in the directory
files = dir(fullfile(folder_path, '*.mat'));

% Preallocate a structure to hold the results
results = struct('FileName', {}, 'SignalType', {}, 'Fs', {}, ...
    'Lag', {}, 'Dimension', {}, 'ExpansionRange', {}, ...
    'LyapunovExponent', {});

fprintf('Starting Lyapunov Exponent computation...\n');
fprintf('--------------------------------------------------------------------------------\n');

for i = 1:length(files)
    filename = files(i).name;
    filepath = fullfile(folder_path, filename);

    % Load the variable 'x' from the .mat file
    data = load(filepath, 'x');
    x = data.x(:);
    N = length(x);

    % Parse the signal type from the filename (e.g., DET_lorenz_x_r01.mat -> lorenz_x)
    first_uscore = find(filename == '_', 1);
    r_tag = regexp(filename, '_r\d+\.mat$', 'once');
    signal_type = filename(first_uscore+1 : r_tag-1);

    % Default parameters (appropriate for 1D/2D discrete maps)
    lag = 1;
    dim = 2;
    exp_range = [1, 5];

    % Assign sampling frequency (Fs) and phase-space reconstruction parameters
    switch signal_type
        case 'lorenz_x'
            fs = 50;
            [~, lag, est_dim] = phaseSpaceReconstruction(x, 'MaxLag', 50);
            dim = max(est_dim, 3);
            exp_range = [20, 150];   % 0.4 s to 3.0 s (linear divergence regime)

        case 'rossler_x'
            fs = 10;
            [~, lag, est_dim] = phaseSpaceReconstruction(x, 'MaxLag', 50);
            dim = max(est_dim, 3);
            exp_range = [50, 300];   % 5.0 s to 30.0 s (linear divergence regime)

        case {'linear', 'exponential'}
            fs = N - 1;

        case {'quadratic', 'polynomial'}
            fs = (N - 1) / 2;

        case {'sinusoid', 'two_tone', 'sawtooth', 'chirp', ...
                'logistic_map', 'henon_map', 'white_gaussian', ...
                'white_uniform', 'random_binary', 'ar1_colored', ...
                'smoothed_noise', 'random_walk'}
            fs = 1;

        otherwise
            warning('Unknown signal type for %s, defaulting Fs to 1.', filename);
            fs = 1;
    end

    % Compute the Lyapunov Exponent
    try
        lyap_exp = lyapunovExponent(x, fs, lag, dim, 'ExpansionRange', exp_range);
    catch ME
        warning('Failed to compute LE for %s: %s', filename, ME.message);
        lyap_exp = NaN;
    end

    % Save the results
    results(i).FileName         = filename;
    results(i).SignalType       = signal_type;
    results(i).Fs               = fs;
    results(i).Lag              = lag;
    results(i).Dimension        = dim;
    results(i).ExpansionRange   = exp_range;
    results(i).LyapunovExponent = lyap_exp;

    fprintf('File: %-26s | Fs: %-6g | Lag: %-2d | Dim: %d | LE: %f\n', ...
        filename, fs, lag, dim, lyap_exp);
end

% Convert the results structure into a table for easy viewing and saving
resultsTable = struct2table(results);

% Optional: Save the results to a CSV file
% writetable(resultsTable, fullfile(folder_path, 'lyapunov_results.csv'));

fprintf('--------------------------------------------------------------------------------\n');
fprintf('Computation finished.\n');