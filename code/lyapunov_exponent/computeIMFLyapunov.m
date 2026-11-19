function lyap = computeIMFLyapunov(IMF, fs)
% COMPUTEIMFLYAPUNOV Computes the largest Lyapunov exponent for one or
% more pre-computed Intrinsic Mode Functions (IMFs).
%
% Inputs:
%   IMF - 1D vector (single IMF) or 2D matrix (each column is an IMF)
%   fs  - (Optional) sampling frequency in Hz. If omitted or empty,
%         defaults to 1, returning exponents in units of [1/sample].
%
% Output:
%   lyap - scalar or 1D row vector of largest Lyapunov exponents
%          Units: [1/s] if fs is provided, [1/sample] if fs is omitted.

if nargin < 2 || isempty(fs)
    fs = 1; % Compute per-sample Lyapunov exponent when fs is unknown
end

% Ensure a single row-vector IMF is converted to a column vector
if isrow(IMF)
    IMF = IMF(:);
end

numIMFs = size(IMF, 2);
lyap = zeros(1, numIMFs);

for k = 1:numIMFs
    current_imf = IMF(:, k);
    N = length(current_imf);

    % Estimate mean oscillation period in SAMPLES via zero-crossings
    % An alternative approach to zero-crossings is also using HHT, to be
    % tested...
    zero_crossings = sum(abs(diff(current_imf > 0)));

    if zero_crossings > 2
        mean_period_samples = N / (zero_crossings / 2);
    else
        mean_period_samples = N;
    end

    % Phase-space parameters in discrete samples (independent of fs)
    embed_lag = max(1, round(mean_period_samples / 4));

    % Cap Theiler window at N/10 so slow, low-frequency IMFs still have
    % enough valid neighbor pairs outside the exclusion window
    theiler_window = max(1, min(round(mean_period_samples), round(N / 10)));

    % Compute Largest Lyapunov Exponent using Rosenstein's algorithm
    try
        lyap(k) = lyapunovExponent(current_imf, fs, ...
            'Lag', embed_lag, ...
            'MinSeparation', theiler_window);
    catch ME
        warning('Failed to compute Lyapunov exponent for IMF %d: %s', k, ME.message);
        lyap(k) = NaN;
    end
end
end