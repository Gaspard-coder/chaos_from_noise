function [H, pValue, Lambda, Orders, CI, allLambda] = chaostest_theta_v3(X, ActiveFN, maxOrders, alpha, flag, dumpfile)
%CHAOSTEST performs the test for Chaos to test the positivity of the
%   dominant Lyapunov Exponent LAMBDA.
%
%   The test hypothesis are:
%   Null hypothesis: LAMBDA >= 0 which indicates the presence of chaos.
%   Alternative hypothesis: LAMBDA < 0 indicates no chaos.
%   This is a one tailed test.
%
%   [H, pValue, LAMBDA, Orders, CI, AllLambda] = ...
%           CHAOSTEST(Series, ActiveFN, maxOrders, ALPHA, FLAG, DUMPFILE)


% Changes: Standardizes X to zero mean and unit variance at startup; 
% computes and prints NMSE alongside SSE per fit; 
% outputs all 5 reconstruction metrics on the standardized series at the 
% end and saves them in dumpfile.

existToolbox = license('test','optimization_toolbox') && ...
    license('test','statistics_toolbox');
if ~existToolbox
    error('The ''CHAOSTEST'' function needs both Optimization and Statistics toolboxes.')
end

if (nargin >= 1) && (~isempty(X))
    if istable(X)
        X = table2array(X);
    end
    if numel(X) == length(X)
        X = X(:);
    else
        error(' Observation ''Series'' must be a vector.');
    end
    X(isnan(X)) = [];

    % Standardize series to zero mean and unit variance (lambda is invariant
    % to affine rescaling, and tanh avoids saturation / scale mismatch)
    muX = mean(X);
    sigmaX = std(X);
    if sigmaX > 0
        X = (X - muX) / sigmaX;
    else
        X = X - muX;
    end
else
    error(' Must enter at least observation vector ''Series''.');
end

if nargin >= 2 && ~isempty(ActiveFN)
    if ~ischar(ActiveFN)
        error(' Activation function ''ActiveFN'' must be a string.')
    end
    if ~any(strcmpi(ActiveFN , {'tanh' , 'logistic' , 'sigmoid'}))
        error('Activaton function ''ActiveFN'' must be TANH, LOGISTIC, or SIGMOID');
    end
else
    ActiveFN = 'tanh';
end

if (nargin >= 3) && ~isempty(maxOrders)
    if numel(maxOrders) ~= 3 || any((maxOrders - round(maxOrders) ~= 0)) || ...
            any((maxOrders <= 0))
        error(' Maximum orders ''maxOrders'' must a vector of 3 positive integers.');
    end
else
    maxOrders = [5, 6, 5];
end
maxL = maxOrders(1);
maxM = maxOrders(2);
maxQ = maxOrders(3);

if length(X) <= maxL * maxM
    error(' Observation ''Series'' must have at least %d obeservations.',maxL*maxM+1)
end

if (nargin >= 4) && ~isempty(alpha)
    if numel(alpha) > 1
        error(' Significance level ''Alpha'' must be a scalar.');
    end
    if (alpha <= 0 || alpha >= 1)
        error(' Significance level ''Alpha'' must be between 0 and 1.');
    end
else
    alpha = 0.05;
end

if nargin >= 5 && ~isempty(flag)
    if ~ischar(flag)
        error(' Regressions type ''FLAG'' must be a string.')
    end
else
    flag = 'VARY';
end
if strcmpi(flag, 'FIX')
    StartL = maxL;
    StartM = maxM;
    StartQ = maxQ;
else
    StartL = 1;
    StartM = 1;
    StartQ = 1;
end

if nargin < 6 || isempty(dumpfile), dumpfile = 'dump.mat'; end

Lambda = -Inf;
Orders = [maxL, maxM, maxQ];
allLambda   = -Inf(maxL, maxM, maxQ);
ThetaAll    = cell(maxL, maxM, maxQ);
ResnormAll  = NaN(maxL, maxM, maxQ);
NMSEAll     = NaN(maxL, maxM, maxQ);
ExitflagAll = NaN(maxL, maxM, maxQ);

Options = optimoptions('lsqnonlin', ...
    'Display', 'off', ...
    'SpecifyObjectiveGradient', true, ...
    'Algorithm', 'levenberg-marquardt', ...
    'StepTolerance', 1e-8, ...
    'FunctionTolerance', 1e-8);

StartValue = 0.5;
nTotal = (maxL-StartL+1) * (maxM-StartM+1) * (maxQ-StartQ+1);
iter   = 0;
tStart = tic;
fprintf('chaostest_theta: %d fits, %d observations (standardized)\n', nTotal, length(X));

for L = StartL:maxL
    for m = StartM:maxM
        for q = StartQ:maxQ
            rng(42); % Reproducible symmetry breaking
            scale = 1 / max(std(X), realmin);
            B = randn(q, m) * scale * 0.5;
            C = randn(q, 1) * 0.5;
            A = randn(1, q) * (std(X) / sqrt(q));
            D = mean(X);

            Theta0 = [reshape(B', 1, q*m), C', A, D];

            [Theta, resnorm, ~, exitflag] = lsqnonlin(@chaosfn, Theta0, [], [], Options, X, L, m, q, ActiveFN);
            T = length(X) - m*L;
            ySub = X(m*L+1:end);
            varSub = mean((ySub - mean(ySub)).^2);
            nmseCur = resnorm / (T * max(varSub, realmin));

            ThetaAll{L,m,q}    = Theta;
            ResnormAll(L,m,q)  = resnorm;
            NMSEAll(L,m,q)     = nmseCur;
            ExitflagAll(L,m,q) = exitflag;

            JacobianMatrix = jacobmatx(Theta, X, L, m, q, ActiveFN);

            h = round(T ^ (1/3));
            newIndex = (h:h:T);
            M = length(newIndex);

            Tn = JacobianMatrix(:, :, newIndex);
            Q_qr = eye(m);
            r_diag_sum = 0;

            % Iterative QR to prevent matrix explosion (Eckmann-Ruelle)
            for t = 1:M
                [Q_qr, R_qr] = qr(Tn(:, :, t) * Q_qr);
                r_diag_sum = r_diag_sum + log(max(abs(R_qr(1, 1)), realmin));

                if t > 1
                    Tn(:, :, t) = Tn(:, :, t) * Tn(:,:,t-1);
                end
            end

            Lambda1 = r_diag_sum / M;
            allLambda(L, m, q) = Lambda1;

            iter = iter + 1;
            el   = toc(tStart);
            fprintf('[%3d/%3d] (L,m,q)=(%d,%d,%d)  SSE=%.4g  NMSE=%.3g  exitflag=%d  lambda=%+.4f  elapsed %4.0fs  ETA %4.0fs\n', ...
                iter, nTotal, L, m, q, resnorm, nmseCur, exitflag, Lambda1, el, el/iter*(nTotal-iter));

            if (Lambda1 > Lambda)
                Change = true;
            elseif (Lambda1 == Lambda) && (L+m+q) < sum(Orders)
                Change = true;
            else
                Change = false;
            end

            if Change
                Lambda = Lambda1;
                Orders = [L, m, q];
                MBest = M;  hBest = h;  TBest = T;
                fprintf('          -> new best lambda = %+.4f at (%d,%d,%d)\n', Lambda, L, m, q);

                Eta  = zeros(M, 1);
                Zeta = Tn(:, :, 1)' * Tn(:, :, 1);
                Zeta = min(realmax, max(realmin, Zeta));

                Eta(1) = 0.5 * log(max(realmin, max(eig(Zeta)))) - Lambda;
                for t = 2:M
                    Zeta1 = min(realmax, max(realmin, Tn(:, :, t)' * Tn(:, :, t)));
                    Zeta2 = min(realmax, max(realmin, Tn(:, :, t-1)' * Tn(:, :, t-1)));
                    Eta(t) = 0.5 * log(max(realmin, max(eig(Zeta1))) ./ ...
                        max(realmin, max(eig(Zeta2)))) - Lambda;
                end
                gamm = zeros(M, 1);
                for i = 1:M
                    gamm(i) = 1/M * Eta(M-i+1:M)' * Eta(1:i);
                end
                gamm        = [gamm(1:end-1); flipud(gamm)];
                Sm          = 1.3221 * (1 * M)^(1/5);
                j           = -M+1:M-1;
                KernelFN    = ones(size(j));
                z           = 6 * pi * (j./Sm) / 5;
                z(j==0)     = [];
                KernelFN(j~=0)  = (3 ./ z.^2) .* (sin(z) ./ z - cos(z));
                varLambda       = max(realmin, KernelFN * gamm);
                LambdaStatistic = Lambda / sqrt(varLambda / M);
            end
        end
    end
end

pValue = normcdf(LambdaStatistic, 0, 1);
crit   = norminv(1 - alpha, 0, 1) * sqrt(varLambda / M);
CI     = [(Lambda - crit), Inf];
H      = (alpha >= pValue);

fprintf('Done in %.0fs. Best (L,m,q)=(%d,%d,%d), lambda=%+.4f, p=%.3g\n', ...
    toc(tStart), Orders(1), Orders(2), Orders(3), Lambda, pValue);

% =========================================================================
% RECONSTRUCTION, PLOTTING, AND DIFFERENCE EVALUATION (STANDARDIZED SCALE)
% =========================================================================
bestL = Orders(1);
bestM = Orders(2);
bestQ = Orders(3);
bestTheta = ThetaAll{bestL, bestM, bestQ};

[residuals, ~] = chaosfn(bestTheta, X, bestL, bestM, bestQ, ActiveFN);
yTarget = X(bestM*bestL+1:end);
yReconstructed = yTarget - residuals';

diffVector  = yTarget - yReconstructed;
sseVal      = sum(diffVector.^2);
varTarget   = mean((yTarget - mean(yTarget)).^2);
nmseVal     = sseVal / (length(yTarget) * max(varTarget, realmin));
rmseVal     = sqrt(mean(diffVector.^2));
meanAbsDiff = mean(abs(diffVector));
maxAbsDiff  = max(abs(diffVector));

fprintf('\n--- Reconstruction Differences ---\n');
fprintf('Sum of Squared Errors (SSE):    %.6g\n', sseVal);
fprintf('Normalized MSE (NMSE):          %.6g\n', nmseVal);
fprintf('Root Mean Squared Error (RMSE): %.6g\n', rmseVal);
fprintf('Mean Absolute Difference:       %.6g\n', meanAbsDiff);
fprintf('Maximum Absolute Difference:    %.6g\n', maxAbsDiff);

figure('Name', 'Chaos Test: Time Series Reconstruction', 'NumberTitle', 'off');
timeIdx = (bestM*bestL+1):length(X);
plot(timeIdx, yTarget, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original Series (standardized)');
hold on;
plot(timeIdx, yReconstructed, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Reconstructed Series');
grid on;
xlabel('Time Index (t)');
ylabel('Series Value (standardized)');
title(sprintf('Original vs Reconstructed Series (Best L=%d, m=%d, q=%d)', bestL, bestM, bestQ));
legend('Location', 'best');
hold off;

save(dumpfile, 'H', 'pValue', 'Lambda', 'CI', 'Orders', 'allLambda', ...
    'ThetaAll', 'ResnormAll', 'NMSEAll', 'ExitflagAll', ...
    'Eta', 'varLambda', 'LambdaStatistic', 'MBest', 'hBest', 'TBest', 'M', ...
    'alpha', 'ActiveFN', 'maxOrders', 'flag', 'StartValue', 'muX', 'sigmaX', ...
    'yTarget', 'yReconstructed', 'sseVal', 'nmseVal', 'rmseVal', 'meanAbsDiff', 'maxAbsDiff');

%-------------------------------------------------------------------------%
function [y, J] = chaosfn(Theta, X, L, m, q, ActiveFN)
B = reshape(Theta(1:q*m), m, q)';
C = Theta(q*m+1:q*m+q)';
A = Theta(q*m+q+1:q*m+q+q);
D = Theta(q*m+2*q+1);

XLAG = lagmatrix(X, L:L:L*m);
XLAG = XLAG(m*L+1:end, :);
T    = length(X) - m*L;

u = B * XLAG' + C;
J = ones(length(Theta), T);

switch upper(ActiveFN)
    case 'LOGISTIC'
        y = D + A * (1 ./ (1 + exp(- u)));
        J(1:q*m,:)           = repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(exp(- u) ./ (1 + exp(- u)).^2, m, 1));
        J(q*m+1:q*m+q,:)     = A' .* (exp(- u) ./ (1 + exp(- u)).^2);
        J(q*m+q+1:q*m+q+q,:) = 1 ./ (1 + exp(- u));

    case 'TANH'
        y = D + A * tanh(u);
        J(1:q*m,:)           = repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(sech(u).^2, m, 1));
        J(q*m+1:q*m+q,:)     = A' .* (sech(u).^2);
        J(q*m+q+1:q*m+q+q,:) = tanh(u);

    case 'SIGMOID'
        y = D + A * (u .* (1 + abs(u / 2)) ./ (2 + abs(u) + u.^2 / 2));
        J(1:q*m,:)           = repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(8*(1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2, m, 1));
        J(q*m+1:q*m+q,:)     = A' .* (8*(1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2);
        J(q*m+q+1:q*m+q+q,:) = u .* (1 + abs(u/2)) ./ (2 + abs(u) + u.^2 / 2);

    otherwise
        error(' Unrecognized activation function!')
end
J = - J';
y = X(m*L+1:end)' - y;

%-------------------------------------------------------------------------%
function YLag = lagmatrix(Y,lags)
if numel(Y) == length(Y)
    Y = Y(:);
end
lags = lags(:);
numLags = length(lags);
[numObs,numSeries] = size(Y);
YLag = nan(numObs,numSeries*numLags);
for i = 1:numLags
    L       = lags(i);
    columns = (numSeries*(i-1)+1):i*numSeries;
    if L > 0
        YLag((L + 1):end,columns) = Y(1:(end - L), :);
    elseif L < 0
        YLag(1:(end + L),columns) = Y((1 - L):end, :);
    else
        YLag(:,columns) = Y;
    end
end

%-------------------------------------------------------------------------%
function J = jacobmatx(Theta, X, L, m, q, ActiveFN)
XLAG = lagmatrix(X, L:L:L*m);
XLAG = XLAG(m*L+1:end, :);
T    = length(X) - m*L;
B = reshape(Theta(1:q*m), m, q)';
C = Theta(q*m+1:q*m+q)';
A = Theta(q*m+q+1:q*m+q+q);

u = B * XLAG' + C;
J = zeros(m, m, T);
J(2:end, 1:end-1, :) = repmat(eye(m-1), [1 1 T]);

switch upper(ActiveFN)
    case 'LOGISTIC'
        J(1, :, :) = reshape(B' * (A' .* (exp(- u) ./ (1 + exp(- u)).^2)), 1, m, T);
    case 'TANH'
        J(1, :, :) = reshape(B' * (A' .* (sech(u).^2)), 1, m, T);
    case 'SIGMOID'
        J(1, :, :) = reshape(B' * (A' .* 8 .* (1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2), 1, m, T);
    otherwise
        error(' Unrecognized activation function!')
end