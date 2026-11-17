function [results, best] = deter_test_sweep(folder_path, m_range, tau_range, eps_range, opts)
% DETER_TEST_SWEEP Sweeps the RQA parameters (m, tau, eps) of deter_test_3
% on a folder of LABELLED signals and finds the best-performing set.
%
%   [results, best] = deter_test_sweep(folder_path, m_range, tau_range, eps_range, opts)
%
%   folder_path : folder with .mat files whose names start with
%                   DET_  (deterministic)  or  RND_  (random/noisy)
%                 (see generate_labeled_signals). The first variable of each
%                 file is used as the signal.
%   m_range     : embedding dimensions to test,   e.g. 2:6
%   tau_range   : time delays to test,            e.g. 1:5
%   eps_range   : thresholds, as a FRACTION of the max distance (same
%                 meaning as in deter_test_3),    e.g. [0.02 0.05 0.1 0.15 0.2]
%   opts        : (optional) struct with fields
%       MaxSamples : use only the first MaxSamples points of each signal.
%                    Default 2000. Cost grows as N^2 (the distance matrix
%                    is N x N): 10000 points needs ~25x more time and ~2 GB
%                    of RAM per matrix. Use Inf for the full length.
%       Threshold  : DET threshold (%) used to classify.       Default 40
%       RankBy     : 'BalancedAccuracy' (default) | 'AUC' | 'OptBalAccuracy'
%       Plot       : true/false                                Default true
%       Verbose    : true/false                                Default true
%
%   Outputs
%     results : table with one row per (m, tau, eps), sorted best-first.
%               Columns:
%                 Accuracy, BalancedAccuracy, TPR (deterministic recall),
%                 TNR (random recall)  -> at the fixed opts.Threshold
%                 AUC                  -> threshold-free separability
%                 OptThreshold, OptBalAccuracy -> best achievable threshold
%                 MeanDET_Det, MeanDET_Rnd     -> mean DET of each class
%                 Margin = min(DET of deterministic) - max(DET of random);
%                          > 0 means a perfect separating threshold exists
%     best    : struct with the winning m, tau, eps, its metrics row and a
%               per-signal table (DET value, true/predicted label).
%
%   The DET computation is a vectorised version of deter_test_3 (identical
%   result, much faster), and the distance matrix is computed once per
%   (signal, m, tau) and reused for every eps. If deter_test_3.m is on the
%   path, the result is cross-checked against it.
%
%   Requires the Statistics and Machine Learning Toolbox (pdist, tiedrank),
%   as deter_test_3 already does.
%
%   Example:
%       generate_labeled_signals('labeled_signals');
%       [results, best] = deter_test_sweep('labeled_signals', 2:6, 1:5, ...
%                                          [0.02 0.05 0.1 0.15 0.2]);
%       deter_test_batch('my_data', best.m, best.tau, best.eps);

%% ---- Options --------------------------------------------------------
defaults = struct('MaxSamples', 2000, 'Threshold', 40, ...
    'RankBy', 'BalancedAccuracy', 'Plot', true, 'Verbose', true);
if nargin < 5 || isempty(opts), opts = struct(); end
fn = fieldnames(defaults);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}), opts.(fn{i}) = defaults.(fn{i}); end
end
thr0 = opts.Threshold;

if any(eps_range <= 0) || any(eps_range > 1)
    error('eps_range values are fractions of the max distance and must be in (0, 1].');
end

%% ---- Load labelled signals -----------------------------------------
files = dir(fullfile(folder_path, '*.mat'));
if isempty(files)
    error('No .mat files found in: %s', folder_path);
end

X = {}; names = {}; isDet = false(0, 1);
for k = 1:numel(files)
    nm = files(k).name;
    if startsWith(nm, 'DET_', 'IgnoreCase', true)
        lab = true;
    elseif startsWith(nm, 'RND_', 'IgnoreCase', true)
        lab = false;
    else
        warning('Skipping "%s": file name must start with DET_ or RND_.', nm);
        continue;
    end

    S = load(fullfile(folder_path, nm));
    v = fieldnames(S);
    x = S.(v{1});
    if ~isnumeric(x) || ~isvector(x) || any(~isfinite(x(:)))
        warning('Skipping "%s": first variable is not a finite 1D numeric vector.', nm);
        continue;
    end
    x = double(x(:));
    if numel(x) > opts.MaxSamples
        x = x(1:opts.MaxSamples);
    end

    X{end+1, 1}     = x;      %#ok<AGROW>
    names{end+1, 1} = nm;     %#ok<AGROW>
    isDet(end+1, 1) = lab;    %#ok<AGROW>
end

nSig = numel(X);
if ~any(isDet) || ~any(~isDet)
    error('Need at least one DET_ and one RND_ signal (found %d DET, %d RND).', ...
        sum(isDet), sum(~isDet));
end
if opts.Verbose
    fprintf('Loaded %d signals (%d deterministic, %d random), up to %d points each.\n', ...
        nSig, sum(isDet), sum(~isDet), max(cellfun(@numel, X)));
end

%% ---- Sweep: DET for every signal x eps x m x tau ---------------------
nM = numel(m_range); nT = numel(tau_range); nE = numel(eps_range);
nPts = cellfun(@numel, X);
DET = nan(nSig, nE, nM, nT);
firstValid = [];
t0 = tic;

for im = 1:nM
    m = m_range(im);
    for it = 1:nT
        tau = tau_range(it);

        if any(nPts - (m - 1) * tau < 20)
            warning('Skipping m=%d, tau=%d: embedding leaves fewer than 20 states.', m, tau);
            continue;
        end
        if isempty(firstValid), firstValid = [im it]; end

        for is = 1:nSig
            D = embed_distances(X{is}, m, tau);   % computed once, reused for all eps
            maxD = max(D(:));
            for ie = 1:nE
                DET(is, ie, im, it) = rqa_det(D <= eps_range(ie) * maxD);
            end
        end

        if opts.Verbose
            fprintf('  m=%d, tau=%d done (%.1f s elapsed)\n', m, tau, toc(t0));
        end
    end
end

%% ---- Cross-check against deter_test_3 (if available) ---------------
if ~isempty(firstValid) && exist('deter_test_3', 'file') == 2
    im0 = firstValid(1); it0 = firstValid(2);
    for is = [find(isDet, 1), find(~isDet, 1)]
        ref = deter_test_3(X{is}, m_range(im0), tau_range(it0), eps_range(1));
        if isnan(ref), ref = 0; end
        if abs(ref - DET(is, 1, im0, it0)) > 1e-9
            warning(['Fast DET (%.6f) differs from deter_test_3 (%.6f) for "%s". ' ...
                'Results may not match deter_test_batch.'], ...
                DET(is, 1, im0, it0), ref, names{is});
        end
    end
end

%% ---- Metrics for every parameter combination -----------------------
nComb = nM * nT * nE;
[Mv, Tv, Ev] = deal(zeros(nComb, 1));
[Acc, BA, TPR, TNR, AUC, OptThr, OptBA, MeanDet, MeanRnd, Margin] = deal(nan(nComb, 1));

row = 0;
for im = 1:nM
    for it = 1:nT
        for ie = 1:nE
            row = row + 1;
            Mv(row) = m_range(im); Tv(row) = tau_range(it); Ev(row) = eps_range(ie);
            s = DET(:, ie, im, it);
            if any(isnan(s)), continue; end      % skipped combination
            q = class_metrics(s, isDet, thr0);
            Acc(row) = q.Acc;   BA(row) = q.BA;   TPR(row) = q.TPR;  TNR(row) = q.TNR;
            AUC(row) = q.AUC;   OptThr(row) = q.OptThr;  OptBA(row) = q.OptBA;
            MeanDet(row) = q.MeanDet;  MeanRnd(row) = q.MeanRnd;  Margin(row) = q.Margin;
        end
    end
end

results = table(Mv, Tv, Ev, Acc, BA, TPR, TNR, AUC, OptThr, OptBA, MeanDet, MeanRnd, Margin, ...
    'VariableNames', {'m', 'tau', 'eps', 'Accuracy', 'BalancedAccuracy', 'TPR', 'TNR', ...
    'AUC', 'OptThreshold', 'OptBalAccuracy', 'MeanDET_Det', 'MeanDET_Rnd', 'Margin'});

%% ---- Rank ------------------------------------------------------------
switch lower(opts.RankBy)
    case 'balancedaccuracy', key = results.BalancedAccuracy;
    case 'auc',              key = results.AUC;
    case 'optbalaccuracy',   key = results.OptBalAccuracy;
    otherwise, error('opts.RankBy must be ''BalancedAccuracy'', ''AUC'' or ''OptBalAccuracy''.');
end
rankMat = permute(reshape(key, [nE nT nM]), [2 1 3]);   % (tau x eps x m), for the heatmaps

% primary metric, then AUC, then margin as tie-breakers
K = [key, results.AUC, results.Margin];
K(isnan(K)) = -Inf;
[~, order] = sortrows(K, [-1 -2 -3]);
results = results(order, :);

%% ---- Best parameter set ---------------------------------------------
b = results(1, :);
best.m = b.m;  best.tau = b.tau;  best.eps = b.eps;
best.metrics = b;

im_b = find(m_range == b.m, 1);
it_b = find(tau_range == b.tau, 1);
ie_b = find(eps_range == b.eps, 1);
detBest = DET(:, ie_b, im_b, it_b);

pred = detBest > thr0;
TrueLabel = repmat({'Random/Noisy'}, nSig, 1);   TrueLabel(isDet) = {'Deterministic'};
Predicted = repmat({'Random/Noisy'}, nSig, 1);   Predicted(pred)  = {'Deterministic'};
best.PerSignal = table(names, TrueLabel, detBest, Predicted, pred == isDet, ...
    'VariableNames', {'FileName', 'TrueLabel', 'DET_Percent', 'Predicted', 'Correct'});

%% ---- Report ----------------------------------------------------------
if opts.Verbose
    fprintf('\nTop parameter sets (ranked by %s, threshold = %g%%):\n', opts.RankBy, thr0);
    disp(results(1:min(10, height(results)), :));
    fprintf('BEST: m = %d, tau = %d, eps = %g  ->  balanced accuracy %.1f%%, AUC %.3f\n', ...
        best.m, best.tau, best.eps, 100 * b.BalancedAccuracy, b.AUC);
    fprintf('      best achievable threshold for these parameters: %.1f%% (balanced accuracy %.1f%%)\n', ...
        b.OptThreshold, 100 * b.OptBalAccuracy);
    wrong = best.PerSignal(~best.PerSignal.Correct, :);
    if ~isempty(wrong)
        fprintf('Misclassified at the best parameters (threshold %g%%):\n', thr0);
        disp(wrong);
    end
end

%% ---- Plots -----------------------------------------------------------
if opts.Plot
    % Heatmaps of the ranking metric over (tau, eps), one panel per m
    figure('Name', 'Parameter sweep', 'Color', 'w');
    nc = min(nM, 3); nr = ceil(nM / nc);
    for im = 1:nM
        subplot(nr, nc, im);
        imagesc(rankMat(:, :, im), [0 1]);
        set(gca, 'YDir', 'normal', ...
            'XTick', 1:nE, 'XTickLabel', arrayfun(@num2str, eps_range, 'UniformOutput', false), ...
            'YTick', 1:nT, 'YTickLabel', arrayfun(@num2str, tau_range, 'UniformOutput', false));
        xlabel('\epsilon (fraction of max distance)'); ylabel('\tau');
        title(sprintf('m = %d', m_range(im)));
        colorbar;
        if m_range(im) == best.m
            hold on; plot(ie_b, it_b, 'kp', 'MarkerFaceColor', 'w', 'MarkerSize', 12); hold off;
        end
    end
    sgtitle(sprintf('%s  (star = best)', opts.RankBy));

    % DET of every signal at the best parameters
    figure('Name', 'DET at best parameters', 'Color', 'w');
    bh = bar(detBest);
    bh.FaceColor = 'flat';
    cd = zeros(nSig, 3);
    cd(isDet, :)  = repmat([0.20 0.60 0.30], nnz(isDet), 1);    % green = deterministic
    cd(~isDet, :) = repmat([0.80 0.30 0.20], nnz(~isDet), 1);   % red   = random
    bh.CData = cd;
    yline(thr0, '--k', sprintf('threshold %g%%', thr0));
    set(gca, 'XTick', 1:nSig, 'XTickLabel', strrep(names, '.mat', ''), ...
        'XTickLabelRotation', 90, 'TickLabelInterpreter', 'none');
    ylabel('DET (%)');
    title(sprintf('Best: m=%d, tau=%d, eps=%g (green = deterministic, red = random)', ...
        best.m, best.tau, best.eps));
end
end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function D = embed_distances(x, m, tau)
% Time-delay embedding + full Euclidean distance matrix (as in deter_test_3)
N  = numel(x);
Ne = N - (m - 1) * tau;
E  = zeros(Ne, m);
for i = 1:m
    E(:, i) = x((1 + (i-1)*tau):(Ne + (i-1)*tau));
end
D = squareform(pdist(E, 'euclidean'));
end

function DET = rqa_det(R)
% Vectorised equivalent of the diagonal-line loop in deter_test_3.
% A recurrence point belongs to a diagonal line of length >= 2 iff its
% up-left or down-right diagonal neighbour is also a recurrence point.
n = size(R, 1);
total = nnz(R) - n;                  % recurrences excluding the main diagonal
if total <= 0
    DET = 0;                         % no off-diagonal recurrences (deter_test_3 gives NaN)
    return;
end
P = false(n + 2);
P(2:n+1, 2:n+1) = R;
hasNeighbour = P(1:n, 1:n) | P(3:n+2, 3:n+2);
inLine = triu(R & hasNeighbour, 1);  % upper triangle only
DET = 100 * (2 * nnz(inLine)) / total;   % x2 for the symmetric lower triangle
end

function q = class_metrics(s, isDet, thr0)
% s: DET values (one per signal), isDet: true labels. Rule: DET > thr -> deterministic.
pred   = s > thr0;
q.TPR  = mean(pred(isDet));
q.TNR  = mean(~pred(~isDet));
q.Acc  = mean(pred == isDet);
q.BA   = (q.TPR + q.TNR) / 2;
q.AUC  = compute_auc(s, isDet);
[q.OptThr, q.OptBA] = best_threshold(s, isDet, thr0);
q.MeanDet = mean(s(isDet));
q.MeanRnd = mean(s(~isDet));
q.Margin  = min(s(isDet)) - max(s(~isDet));
end

function auc = compute_auc(s, isDet)
% Rank-based (Mann-Whitney) AUC: P(DET of a deterministic signal > DET of a random one)
pos = s(isDet);  neg = s(~isDet);
r = tiedrank([pos; neg]);
auc = (sum(r(1:numel(pos))) - numel(pos) * (numel(pos) + 1) / 2) / (numel(pos) * numel(neg));
end

function [thr, ba] = best_threshold(s, isDet, thr0)
% Threshold maximising balanced accuracy; ties resolved toward thr0.
u = unique(s);
cand = [u(1) - 1; (u(1:end-1) + u(2:end)) / 2; u(end) + 1];
baVec = zeros(size(cand));
for i = 1:numel(cand)
    pred = s > cand(i);
    baVec(i) = 0.5 * (mean(pred(isDet)) + mean(~pred(~isDet)));
end
ba  = max(baVec);
idx = find(baVec >= ba - 1e-12);
[~, j] = min(abs(cand(idx) - thr0));
thr = cand(idx(j));
end