function [imf, residual, info] = VMD_gen(data, folder, varargin)
% VMD_GEN Performs Variational Mode Decomposition (VMD) on a time-series signal
% and saves all resulting IMFs (modes) and residual components to .mat files.
%
% Syntax:
%   VMD_gen(data, folder)
%   VMD_gen(data, folder, Name=Value)
%   [imf, residual, info] = VMD_gen(...)
%
% Inputs:
%   data     - 1D vector containing the time-series signal.
%   folder   - Target directory path (string or char) to save .mat files.
%
% Optional Name-Value Arguments (for MATLAB's vmd):
%   NumIMFs           - Number of modes (IMFs) to extract (K). (default: 5)
%   PenaltyFactor     - Data-fidelity penalty factor (Alpha). Lower values
%                       yield wider bandwidth modes. (default: 1000)
%   LMUpdateRate      - Lagrangian multiplier update rate (Tau). (default: 0.01)
%   InitializeMethod  - How center frequencies are initialized: 'peaks',
%                       'random', or 'grid'. (default: 'peaks')
%   AbsoluteTolerance - Absolute convergence tolerance. (default: 1e-5)
%   RelativeTolerance - Relative convergence tolerance. (default: 1e-5)
%   MaxIterations     - Maximum optimization iterations. (default: 500)
%   Display           - Toggle optimization diagnostic table: 0 or 1. (default: 0)
%   SampleRate        - Sampling frequency of signal in Hz.
%
% Outputs:
%   imf      - Matrix of extracted IMFs (columns correspond to modes).
%   residual - Residual signal vector.
%   info     - Diagnostic structure returned by vmd (convergence, frequencies, etc.).

% Validate time-series vector format
if ~isvector(data)
    error('Input data must be a 1D time-series vector.');
end
data = data(:); % Ensure column vector format

% Create target directory if it does not exist
if ~exist(folder, 'dir')
    mkdir(folder);
end

% Perform Variational Mode Decomposition using passed Name-Value options
[imf, residual, info] = vmd(data, varargin{:});
fprintf('Number of IMFs extracted: %d\n', size(imf, 2));
disp(info)

% Save individual IMF .mat files
numIMFs = size(imf, 2);
for k = 1:numIMFs
    current_imf = imf(:, k);
    imf_filename = fullfile(folder, sprintf('IMF_%d.mat', k));
    save(imf_filename, 'current_imf');
end

% Save residual .mat file
residual_filename = fullfile(folder, 'residual.mat');
save(residual_filename, 'residual');

% Save complete decomposition output in a single file for convenience
master_filename = fullfile(folder, 'VMD_all.mat');
save(master_filename, 'imf', 'residual', 'info');

fprintf('VMD complete. Saved %d IMFs and residual to: %s\n', numIMFs, folder);
end