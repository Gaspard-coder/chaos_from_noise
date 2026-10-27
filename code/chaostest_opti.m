function [H, pValue, Lambda, Orders, CI, allLambda] = chaostest_opti(X, ActiveFN, maxOrders, alpha, flag)
%CHAOSTEST performs the test for Chaos to test the positivity of the
%   dominant Lyapunov Exponent LAMBDA (Optimized Version).

% Check if Optimization and Statistics toolboxes are installed
existToolbox = license('test','optimization_toolbox') && ...
    license('test','statistics_toolbox');
if ~existToolbox
    error('The ''CHAOSTEST'' function needs both Optimization and Statistics toolboxes.');
end

% Input validation and formatting
if (nargin >= 1) && (~isempty(X))
    if istable(X)
        X = table2array(X);
    end
    if numel(X) == length(X)
        X = X(:); % Column vector
    else
        error('Observation ''Series'' must be a vector.');
    end
    X(isnan(X)) = []; % Remove NaNs
else
    error('Must enter at least observation vector ''Series''.');
end

% Activation function parameter
if nargin >= 2 && ~isempty(ActiveFN)
    if ~ischar(ActiveFN) && ~isstring(ActiveFN)
        error('Activation function ''ActiveFN'' must be a string.');
    end
    ActiveFN = upper(char(ActiveFN));
    if ~any(strcmp(ActiveFN, {'TANH', 'LOGISTIC', 'SIGMOID'}))
        error('Activation function ''ActiveFN'' must be TANH, LOGISTIC, or SIGMOID');
    end
else
    ActiveFN = 'TANH';
end

% Model orders maxOrders = [L, m, q]
if (nargin >= 3) && ~isempty(maxOrders)
    if numel(maxOrders) ~= 3 || any((maxOrders - round(maxOrders) ~= 0)) || any(maxOrders <= 0)
        error('Maximum orders ''maxOrders'' must be a vector of 3 positive integers.');
    end
else
    maxOrders = [5, 6, 5];
end
maxL = maxOrders(1);
maxM = maxOrders(2);
maxQ = maxOrders(3);

if length(X) <= maxL * maxM
    error('Observation ''Series'' must have at least %d observations.', maxL * maxM + 1);
end

% Significance level alpha
if (nargin >= 4) && ~isempty(alpha)
    if numel(alpha) > 1 || alpha <= 0 || alpha >= 1
        error('Significance level ''Alpha'' must be a scalar between 0 and 1.');
    end
else
    alpha = 0.05;
end

% Flag strategy ('VARY' vs 'FIX')
if nargin >= 5 && ~isempty(flag)
    if ~ischar(flag) && ~isstring(flag)
        error('Regression type ''FLAG'' must be a string.');
    end
    flag = upper(char(flag));
else
    flag = 'VARY';
end

if strcmp(flag, 'FIX')
    StartL = maxL; StartM = maxM; StartQ = maxQ;
else
    StartL = 1; StartM = 1; StartQ = 1;
end

% Configure solver options once
Options = optimset('lsqnonlin');
Options = optimset(Options, 'Display', 'off', 'Jacobian', 'on');
StartValue = 0.5;

allLambda = -Inf(maxL, maxM, maxQ);

% Build grid of parameter triplets (L, m, q)
gridL = StartL:maxL;
gridM = StartM:maxM;
gridQ = StartQ:maxQ;

[L_mesh, M_mesh, Q_mesh] = ndgrid(gridL, gridM, gridQ);
numModels = numel(L_mesh);

L_vec = L_mesh(:);
M_vec = M_mesh(:);
Q_vec = Q_mesh(:);

lambdas_computed = -Inf(numModels, 1);
thetas_computed  = cell(numModels, 1);

% Run parallelized or serial loop over model grid
parfor idx = 1:numModels
    L_val = L_vec(idx);
    m_val = M_vec(idx);
    q_val = Q_vec(idx);

    % Precompute XLAG once per iteration
    XLAG = fast_lagmatrix(X, L_val, m_val);
    T_val = length(X) - m_val * L_val;
    X_tail = X(m_val * L_val + 1 : end);

    % Initial parameters
    A = StartValue * ones(q_val, m_val);
    B = StartValue * ones(q_val, 1);
    C = StartValue * ones(1, q_val);
    D = StartValue;
    Theta0 = [reshape(A', 1, q_val * m_val), B', C, D];

    % Optimize parameters using lsqnonlin
    Theta = lsqnonlin(@(th) chaosfn_fast(th, X_tail, XLAG, q_val, ActiveFN), ...
        Theta0, [], [], Options);

    % Compute Jacobian matrix relative to X
    JacobianMatrix = jacobmatx_fast(Theta, XLAG, q_val, ActiveFN);

    % Block subsampling
    h = round(T_val ^ (1/3));
    newIndex = (h : h : T_val);
    M_sub = length(newIndex);

    % Product of Jacobians across blocks
    Tn = JacobianMatrix(:, :, newIndex);
    for t = 2:M_sub
        Tn(:, :, t) = Tn(:, :, t) * Tn(:, :, t-1);
    end

    U0 = [1; zeros(m_val - 1, 1)];
    v = eig(min(realmax, (Tn(:, :, M_sub) * U0)' * (Tn(:, :, M_sub) * U0)));

    Lambda1 = 1 / (2 * M_sub) * log(max([v; realmin]));

    lambdas_computed(idx) = Lambda1;
    thetas_computed{idx}  = Theta;
end

% Populate allLambda matrix and determine best triplet
Lambda = -Inf;
Orders = [maxL, maxM, maxQ];
bestIdx = 1;

for idx = 1:numModels
    L_val = L_vec(idx);
    m_val = M_vec(idx);
    q_val = Q_vec(idx);
    L1 = lambdas_computed(idx);

    allLambda(L_val, m_val, q_val) = L1;

    if (L1 > Lambda) || ((L1 == Lambda) && (L_val + m_val + q_val) < sum(Orders))
        Lambda  = L1;
        Orders  = [L_val, m_val, q_val];
        bestIdx = idx;
    end
end

% Final calculation of asymptotic variance for winning model
L_best = Orders(1);
m_best = Orders(2);
q_best = Orders(3);
Theta_best = thetas_computed{bestIdx};

XLAG_best = fast_lagmatrix(X, L_best, m_best);
T_best = length(X) - m_best * L_best;

JacobianMatrix_best = jacobmatx_fast(Theta_best, XLAG_best, q_best, ActiveFN);

h_best = round(T_best ^ (1/3));
newIndex_best = (h_best : h_best : T_best);
M_best = length(newIndex_best);

Tn_best = JacobianMatrix_best(:, :, newIndex_best);
for t = 2:M_best
    Tn_best(:, :, t) = Tn_best(:, :, t) * Tn_best(:, :, t-1);
end

Eta = zeros(M_best, 1);
Zeta = min(realmax, max(realmin, Tn_best(:, :, 1)' * Tn_best(:, :, 1)));
Eta(1) = 0.5 * log(max(realmin, max(eig(Zeta)))) - Lambda;

for t = 2:M_best
    Zeta1 = min(realmax, max(realmin, Tn_best(:, :, t)' * Tn_best(:, :, t)));
    Zeta2 = min(realmax, max(realmin, Tn_best(:, :, t-1)' * Tn_best(:, :, t-1)));
    Eta(t) = 0.5 * log(max(realmin, max(eig(Zeta1))) ./ max(realmin, max(eig(Zeta2)))) - Lambda;
end

gamm = zeros(M_best, 1);
for i = 1:M_best
    gamm(i) = 1 / M_best * Eta(M_best - i + 1 : M_best)' * Eta(1 : i);
end
gamm = [gamm(1:end-1); flipud(gamm)];

Sm = 1.3221 * (M_best)^(1/5);
j_idx = -M_best + 1 : M_best - 1;
KernelFN = ones(size(j_idx));
z = 6 * pi * (j_idx ./ Sm) / 5;
z(j_idx == 0) = [];
KernelFN(j_idx ~= 0) = (3 ./ z.^2) .* (sin(z) ./ z - cos(z));

varLambda = max(realmin, KernelFN * gamm);
LambdaStatistic = Lambda / sqrt(varLambda / M_best);

pValue = normcdf(LambdaStatistic, 0, 1);

if nargout >= 5
    crit = norminv(1 - alpha, 0, 1) * sqrt(varLambda / M_best);
    CI = [(Lambda - crit), Inf];
end

H = (alpha >= pValue);
end

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function XLAG = fast_lagmatrix(X, L, m)
% FAST_LAGMATRIX Precomputes lagged time-series matrix efficiently.
N = length(X);
T = N - m * L;
XLAG = zeros(T, m);
for i = 1:m
    lag = i * L;
    XLAG(:, i) = X(m * L + 1 - lag : N - lag);
end
end

function [y, J] = chaosfn_fast(Theta, X_tail, XLAG, q, ActiveFN)
% CHAOSFN_FAST Evaluates neural network objective function and analytical Jacobian.
m = size(XLAG, 2);
T = size(XLAG, 1);

A = reshape(Theta(1 : q * m), m, q)'; % [q, m]
B = Theta(q * m + 1 : q * m + q)';    % [q, 1]
C = Theta(q * m + q + 1 : q * m + 2 * q); % [1, q]
D = Theta(q * m + 2 * q + 1);          % scalar

u = A * XLAG' + B; % Implicit expansion: [q, T]

J = ones(length(Theta), T);

switch ActiveFN
    case 'LOGISTIC'
        f_u = 1 ./ (1 + exp(-u));
        dF_du = C' .* (f_u .* (1 - f_u));
        y_pred = D + C * f_u;

    case 'TANH'
        f_u = tanh(u);
        dF_du = C' .* (sech(u).^2);
        y_pred = D + C * f_u;

    case 'SIGMOID'
        f_u = u .* (1 + abs(u / 2)) ./ (2 + abs(u) + u.^2 / 2);
        dF_du = C' .* (8 * (1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2);
        y_pred = D + C * f_u;
end

J(1 : q * m, :) = repmat(dF_du, m, 1) .* repmat(XLAG', q, 1);
J(q * m + 1 : q * m + q, :) = dF_du;
J(q * m + q + 1 : q * m + 2 * q, :) = f_u;

J = -J';
y = X_tail - y_pred';
end

function J = jacobmatx_fast(Theta, XLAG, q, ActiveFN)
% JACOBMATX_FAST Computes Jacobian matrix relative to state X.
m = size(XLAG, 2);
T = size(XLAG, 1);

A = reshape(Theta(1 : q * m), m, q)';
B = Theta(q * m + 1 : q * m + q)';
C = Theta(q * m + q + 1 : q * m + 2 * q);

u = A * XLAG' + B; % Implicit expansion

J = zeros(m, m, T);
for i = 1:m-1
    J(i + 1, i, :) = 1;
end

switch ActiveFN
    case 'LOGISTIC'
        f_u = 1 ./ (1 + exp(-u));
        dF_du = C' .* (f_u .* (1 - f_u));
        J(1, :, :) = reshape(A' * dF_du, [1, m, T]);

    case 'TANH'
        dF_du = C' .* (sech(u).^2);
        J(1, :, :) = reshape(A' * dF_du, [1, m, T]);

    case 'SIGMOID'
        dF_du = C' .* (8 * (1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2);
        J(1, :, :) = reshape(A' * dF_du, [1, m, T]);
end
end