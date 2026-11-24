function [lambda, params, diagData] = estimateLyapunov(x, options)
% ESTIMATELYAPUNOV Estimates the Largest Lyapunov Exponent (LLE) from a 1D
% time series without requiring a sampling frequency (units: 1/sample).
%
% Implements:
%   1. Autocorrelation decay (1/e) for time delay (tau)
%   2. Kennel's False Nearest Neighbors (FNN) for embedding dimension (m)
%   3. Rosenstein's algorithm with a Theiler window for LLE estimation
%
% Usage:
%   [lambda, params, diagData] = estimateLyapunov(x)
%   [lambda, params, diagData] = estimateLyapunov(x, 'Plot', true, 'MaxDim', 12)
%
% Inputs:
%   x       - 1D numeric vector (time series)
%   options - Name-Value pairs:
%               'MaxLag'      (default: 200)
%               'MaxDim'      (default: 10)
%               'FNNThresh'   (default: 5)   - Percentage threshold for FNN
%               'FitFraction' (default: 0.35)- Fraction of max steps for linear fit
%               'Plot'        (default: true)
%
% Outputs:
%   lambda   - Largest Lyapunov Exponent [1/sample]
%   params   - Struct containing estimated tau, m, theilerWindow, and maxSteps
%   diagData - Struct containing curves (acf, fnnPct, steps, divergence, fitCoeffs)

arguments
    x (:,1) double {mustBeNonempty, mustBeFinite}
    options.MaxLag (1,1) double {mustBeInteger, mustBePositive} = 200
    options.MaxDim (1,1) double {mustBeInteger, mustBeGreaterThan(options.MaxDim, 2)} = 10
    options.FNNThresh (1,1) double {mustBePositive} = 5
    options.FitFraction (1,1) double {mustBeInRange(options.FitFraction, 0.05, 1)} = 0.35
    options.Plot (1,1) logical = true
end

%% 1. Signal Normalization
sigma = std(x);
if sigma == 0
    error('Input signal is constant (zero variance). Cannot reconstruct phase space.');
end
x = (x - mean(x)) / sigma;
N = length(x);

%% 2. Automatic Time Delay (tau) via Autocorrelation (1/e threshold)
maxLag = min(floor(N / 10), options.MaxLag);
acf = zeros(maxLag + 1, 1);
for lag = 0:maxLag
    acf(lag + 1) = mean(x(1:N-lag) .* x(1+lag:N));
end

tau = find(acf <= exp(-1), 1, 'first') - 1;
if isempty(tau) || tau < 1
    tau = 1; % Fallback for rapidly decorrelating discrete maps
end

%% 3. Embedding Dimension (m) via False Nearest Neighbors (FNN)
R_tol = 15;     % Kennel et al. (1992) relative distance threshold
A_tol = 2;      % Kennel et al. (1992) attractor size threshold
R_A   = 1;      % std(x) is 1 after normalization

dimsTested = 1:(options.MaxDim - 1);
fnnPct = NaN(size(dimsTested)); % Initialize as NaN to prevent false triggers on early break

for idxDim = 1:length(dimsTested)
    mTest = dimsTested(idxDim);
    MTest = N - mTest * tau;
    if MTest < 50
        break;
    end

    % Vectorized delay embedding matrix [MTest x mTest]
    embIdx = (1:MTest)' + (0:mTest-1) * tau;
    Xm = x(embIdx);

    % Subsample up to 1000 query points for computational efficiency
    nSamples = min(MTest, 1000);
    sampleIdx = round(linspace(1, MTest, nSamples));
    XQuery = Xm(sampleIdx, :);

    % Fast pairwise Euclidean distances [nSamples x MTest]
    distMat = computePairwiseDist(XQuery, Xm);

    % Mask self-matches along the sampled indices
    linSelfIdx = sub2ind(size(distMat), 1:nSamples, sampleIdx);
    distMat(linSelfIdx) = inf;

    [d_m, nnIdx] = min(distMat, [], 2);
    valid = d_m > 0;

    % Evaluate distance growth in dimension mTest + 1
    newCoordDiff = abs(x(sampleIdx(valid)' + mTest * tau) - x(nnIdx(valid) + mTest * tau));
    d_m = d_m(valid);
    d_m1 = hypot(d_m, newCoordDiff);

    isFalse = ((newCoordDiff ./ d_m) > R_tol) | ((d_m1 ./ R_A) > A_tol);
    fnnPct(idxDim) = (sum(isFalse) / sum(valid)) * 100;
end

m = find(fnnPct < options.FNNThresh, 1, 'first');
if isempty(m)
    [~, m] = min(fnnPct, [], 'omitnan');
end
m = max(m, 2); % Enforce minimum embedding dimension of 2

%% 4. Rosenstein's Divergence Calculation with Theiler Window
theilerWindow = max(10, 3 * tau);
maxSteps      = max(30, 5 * tau);

M      = N - (m - 1) * tau;
MValid = M - maxSteps;

if MValid <= theilerWindow
    error('Time series too short (N=%d) for estimated parameters (tau=%d, m=%d).', N, tau, m);
end

% Full phase space reconstruction [M x m]
X = x((1:M)' + (0:m-1) * tau);
XValid = X(1:MValid, :);

% Compute full pairwise distance matrix for valid initial points [MValid x MValid]
distFull = computePairwiseDist(XValid, XValid);

% Apply Theiler window mask using Toeplitz structure (excludes |i - j| <= theilerWindow)
idxVec = 1:MValid;
theilerMask = abs(idxVec' - idxVec) <= theilerWindow;
distFull(theilerMask) = inf;

[dMin, nnIdx] = min(distFull, [], 2);
validPairs = find(dMin > 0 & isfinite(dMin));

% Vectorized tracking of trajectory divergence across all steps k = 0:maxSteps
nValid = length(validPairs);
divergence = NaN(nValid, maxSteps + 1);

for k = 0:maxSteps
    diffVec = X(validPairs + k, :) - X(nnIdx(validPairs) + k, :);
    dk = sqrt(sum(diffVec.^2, 2));
    posMask = dk > 0;
    divergence(posMask, k + 1) = log(dk(posMask));
end

y_k   = mean(divergence, 1, 'omitnan');
steps = 0:maxSteps;

%% 5. Linear Regression over Initial Scaling Region
kStart = 1;
kEnd   = max(5, round(maxSteps * options.FitFraction));

fitMask   = (steps >= kStart) & (steps <= kEnd) & isfinite(y_k);
fitCoeffs = polyfit(steps(fitMask), y_k(fitMask), 1);
lambda    = fitCoeffs(1);

%% 6. Package Outputs & Optional Diagnostics
params = struct( ...
    'tau', tau, ...
    'm', m, ...
    'theilerWindow', theilerWindow, ...
    'maxSteps', maxSteps);

diagData = struct( ...
    'acf', acf, ...
    'dimsTested', dimsTested, ...
    'fnnPct', fnnPct, ...
    'steps', steps, ...
    'divergence', y_k, ...
    'fitMask', fitMask, ...
    'fitCoeffs', fitCoeffs);

if options.Plot
    plotDiagnostics(lambda, params, diagData, options.FNNThresh);
end
end

%% Helper Function: Fast Pairwise Euclidean Distance
function D = computePairwiseDist(A, B)
% Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a*b') for rapid BLAS matrix multiplication
sqA = sum(A.^2, 2);
sqB = sum(B.^2, 2)';
D2 = max(0, sqA + sqB - 2 * (A * B'));
D = sqrt(D2);
end

%% Helper Function: Diagnostic Visualization
function plotDiagnostics(lambda, params, diagData, fnnThresh)
fprintf('--- AUTOMATICALLY ESTIMATED PARAMETERS ---\n');
fprintf('Optimal Time Delay (tau):     %d samples\n', params.tau);
fprintf('Embedding Dimension (m):      %d\n', params.m);
fprintf('Theiler Window:               %d samples\n', params.theilerWindow);
fprintf('------------------------------------------\n');
fprintf('Largest Lyapunov Exponent:    %.4f [1/sample]\n', lambda);

figure('Color', 'w', 'Position', [100, 100, 1000, 400]);

% Subplot 1: False Nearest Neighbors
subplot(1, 2, 1);
plot(diagData.dimsTested, diagData.fnnPct, 'ko-', 'LineWidth', 1.5); hold on;
yline(fnnThresh, 'r--', sprintf('Threshold (%g%%)', fnnThresh));
plot(params.m, diagData.fnnPct(params.m), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
grid on;
xlabel('Test Dimension (m)');
ylabel('False Nearest Neighbors (%)');
title(sprintf('Embedding Dimension Selection (m = %d)', params.m));

% Subplot 2: Divergence Curve & Linear Fit
subplot(1, 2, 2);
plot(diagData.steps, diagData.divergence, 'bo-', 'LineWidth', 1.2, 'MarkerSize', 4); hold on;
fitSteps = diagData.steps(diagData.fitMask);
plot(fitSteps, polyval(diagData.fitCoeffs, fitSteps), 'r--', 'LineWidth', 2.5);
grid on;
xlabel('Discrete Steps (k) [samples]');
ylabel('Mean Divergence \langle ln(d_k) \rangle');
title(sprintf('Lyapunov Exponent: \\lambda = %.4f sample^{-1}', lambda));
legend('Divergence curve', 'Linear scaling fit', 'Location', 'southeast');
end
