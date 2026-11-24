function [results, allDetails] = LP_FNN(folderpath, options)
% LP_FNN Applies estimateLyapunov.m to every .mat file in a specified folder.
%
% Usage:
%   results = LP_FNN('C:\data\signals')
%   [results, details] = LP_FNN('C:\data\signals', 'VarName', 'x', 'Plot', false)
%
% Inputs:
%   folderpath - Path to the directory containing .mat files
%   options    - Name-Value pairs:
%                  'VarName'     (default: "")    - Specific variable name to load.
%                                                   If empty, loads the first 1D numeric vector found.
%                  'Recursive'   (default: false) - Search subfolders recursively
%                  'MaxLag'      (default: 200)   - Passed to estimateLyapunov
%                  'MaxDim'      (default: 10)    - Passed to estimateLyapunov
%                  'FNNThresh'   (default: 5)     - Passed to estimateLyapunov
%                  'FitFraction' (default: 0.35)  - Passed to estimateLyapunov
%                  'Plot'        (default: false) - Disabled by default for batch processing
%
% Outputs:
%   results    - Table summarizing FileName, Lambda, Tau, M, TheilerWindow, MaxSteps, and Status
%   allDetails - Struct array containing full params and diagData for each processed file

arguments
    folderpath (1,1) string {mustBeFolder}
    options.VarName (1,1) string = ""
    options.Recursive (1,1) logical = false
    options.MaxLag (1,1) double {mustBeInteger, mustBePositive} = 200
    options.MaxDim (1,1) double {mustBeInteger, mustBeGreaterThan(options.MaxDim, 2)} = 10
    options.FNNThresh (1,1) double {mustBePositive} = 5
    options.FitFraction (1,1) double {mustBeInRange(options.FitFraction, 0.05, 1)} = 0.35
    options.Plot (1,1) logical = false
end

%% 1. Locate .mat Files
if options.Recursive
    matFiles = dir(fullfile(folderpath, '**', '*.mat'));
else
    matFiles = dir(fullfile(folderpath, '*.mat'));
end

nFiles = numel(matFiles);
if nFiles == 0
    warning('No .mat files found in: %s', folderpath);
    results = table();
    allDetails = struct([]);
    return;
end

%% 2. Preallocate Output Containers
FileName      = strings(nFiles, 1);
Lambda        = NaN(nFiles, 1);
Tau           = NaN(nFiles, 1);
M             = NaN(nFiles, 1);
TheilerWindow = NaN(nFiles, 1);
MaxSteps      = NaN(nFiles, 1);
Status        = strings(nFiles, 1);
ErrorMessage  = strings(nFiles, 1);

allDetails = repmat(struct('FileName', "", 'params', [], 'diagData', []), nFiles, 1);

%% 3. Batch Processing Loop
fprintf('Processing %d .mat file(s) in %s...\n', nFiles, folderpath);

for i = 1:nFiles
    filePath = fullfile(matFiles(i).folder, matFiles(i).name);
    FileName(i) = string(matFiles(i).name);
    allDetails(i).FileName = FileName(i);

    try
        % Extract signal vector from .mat file
        x = extractSignal(filePath, options.VarName);

        % Run Lyapunov estimation
        [lam, params, diagData] = estimateLyapunov(x, ...
            'MaxLag',      options.MaxLag, ...
            'MaxDim',      options.MaxDim, ...
            'FNNThresh',   options.FNNThresh, ...
            'FitFraction', options.FitFraction, ...
            'Plot',        options.Plot);

        % Store results
        Lambda(i)        = lam;
        Tau(i)           = params.tau;
        M(i)             = params.m;
        TheilerWindow(i) = params.theilerWindow;
        MaxSteps(i)      = params.maxSteps;
        Status(i)        = "OK";

        allDetails(i).params   = params;
        allDetails(i).diagData = diagData;

        fprintf('[%d/%d] %s -> lambda = %.4f (tau=%d, m=%d)\n', ...
            i, nFiles, FileName(i), lam, params.tau, params.m);

    catch ME
        Status(i)       = "ERROR";
        ErrorMessage(i) = string(ME.message);
        fprintf('[%d/%d] %s -> FAILED: %s\n', i, nFiles, FileName(i), ME.message);
    end
end

%% 4. Compile Summary Table
results = table(FileName, Lambda, Tau, M, TheilerWindow, MaxSteps, Status, ErrorMessage);
end

%% Helper Function: Extract 1D Signal from .mat File
function x = extractSignal(filePath, varName)
data = load(filePath);
fields = fieldnames(data);

if varName ~= ""
    if ~isfield(data, varName)
        error('Variable "%s" not found in file.', varName);
    end
    raw = data.(varName);
else
    % Automatically find the first numeric vector
    raw = [];
    for k = 1:numel(fields)
        val = data.(fields{k});
        if isnumeric(val) && isvector(val) && numel(val) > 10
            raw = val;
            break;
        end
    end
    if isempty(raw)
        error('No valid 1D numeric vector found in file.');
    end
end

if ~isnumeric(raw) || ~isvector(raw)
    error('Extracted variable is not a 1D numeric vector.');
end

x = double(raw(:)); % Ensure column vector of doubles
end