function [imf, residual, info] = EMD_gen(data, folder, varargin)
% EMD_GEN Performs Empirical Mode Decomposition (EMD) on a time-series signal
% and saves all resulting IMFs and residual components to .mat files.
%
% Syntax:
%   EMD_gen(data, folder)
%   EMD_gen(data, folder, Name=Value)
%   [imf, residual, info] = EMD_gen(...)
%
% Inputs:
%   data     - 1D vector containing the time-series signal.
%   folder   - Target directory path (string or char) to save .mat files.
%
% Optional Name-Value Arguments (for MATLAB's emd):
%   SiftRelativeTolerance - Convergence criterion threshold (default: 0.2).
%   SiftMaxIterations     - Maximum sifting iterations per IMF (default: 100).
%   MaxNumIMF             - Maximum number of IMFs to extract (default: 10).
%   MaxNumExtrema         - Extrema count in residual to stop sifting (default: 1).
%   MaxEnergyRatio        - Maximum signal-to-residual energy ratio (default: 20).
%   Interpolation         - Envelope interpolation method: "spline" or "pchip".
%   Display               - Toggle sifting diagnostic table: 0 or 1 (default: 0).
%   SampleRate            - Sampling frequency of signal in Hz.
%
% Outputs:
%   imf      - Matrix of extracted IMFs (columns correspond to IMFs).
%   residual - Residual signal vector.
%   info     - Diagnostic structure returned by emd.

% Validate time-series vector format
if ~isvector(data)
    error('Input data must be a 1D time-series vector.');
end
data = data(:); % Ensure column vector format

% Create target directory if it does not exist
if ~exist(folder, 'dir')
    mkdir(folder);
end

% Perform Empirical Mode Decomposition using passed Name-Value options
[imf, residual, info] = emd(data, varargin{:});

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
master_filename = fullfile(folder, 'EMD_all.mat');
save(master_filename, 'imf', 'residual', 'info');

fprintf('EMD complete. Saved %d IMFs and residual to: %s\n', numIMFs, folder);
end