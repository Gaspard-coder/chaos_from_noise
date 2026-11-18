function [H, pValue, Lambda, Orders, CI, allLambda] = chaostest_theta_v2(X, ActiveFN, maxOrders, alpha, flag, dumpfile)
%CHAOSTEST performs the test for Chaos to test the positivity of the
%   dominant Lyapunov Exponent LAMBDA.
%
%   The test hypothesis are:
%   Null hypothesis: LAMBDA >= 0 which indicates the presence of chaos.
%   Alternative hypothesis: LAMBDA < 0 indicates no chaos.
%   This is a one tailed test.
%
%   [H, pValue, LAMBDA, Orders, CI, AllLambda] = ...
%           CHAOSTEST(Series, ActiveFN, maxOrders, ALPHA, FLAG)
%
% Inputs:
%   Series - a vector of observation to test.
%
% Optional inputs:
%   ActiveFN - String containing the activation function to use in the
%     neural net estimation. ActiveFN can be the 'LOGISTIC' function
%     f(u) = 1 / (1 + exp(- u)), domain = [0, 1], or 'TANH' function
%     f(u) = tanh(u), domain = [-1, 1], or 'SIGMOID' function
%     f(u) = u * (1 + |u/2|) / (2 + |u| + u^2 / 2), domain = [-1, 1].
%     Default = 'TANH'.
%
%   maxOrders - the maximum orders that the chaos function defined
%     in CHAOSFN can take. This must be a vector containing 3 elements.
%     maxOrders = [L, m, q].
%     Increasing the model's orders can slow down calculations.
%     Default = [5, 6, 5].
%
%   ALPHA - The significance level for the test (default = 0.05)
%
%   FLAG  - String specifying the method to carry out the test, by
%     varying the triplet (L, m, q) {'VARY' or anything else} or by
%     fixing them {'FIX'}. Default = {'VARY'}.
%
%   dumpfile - Dumpfile storing the return values + the intermdiate values
%   of theta and Sum of Squared Errors (SSE)
%
% Outputs:
%   H = 0 => Do not reject the null hypothesis of Chaos at significance
%            level ALPHA.
%   H = 1 => Reject the null hypothesis of Chaos at significance level
%            ALPHA.
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small
%     values of pValue cast doubt on the validity of the null hypothesis
%     of Chaos.
%
%  LAMBDA - The dominant Lyapunov Exponent.
%     If LAMBDA is positive, this indicates the presence of Chaos.
%     If LAMBDA is negative, this indicates the absence of Chaos.
%
%   Orders - gives the triplet (L, m, q) that maximizes the Lyapunov
%     exponent computed from all L*m*q estimations.
%
%   CI - Confidence interval for LAMBDA at level ALPHA.
%
%   allLambda - all the computed Lyapunov exponents for all L-by-m-by-q
%   estimations.
%
%   The algorithm uses the Jacobian method in contrast to the direct
%   method, it needs the optimazition and the statistics toolboxes.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed BenSaïda           %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                   $ Revision 5.1 $ Date: 2 April 2016 $                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if the Optimization and Statistics toolboxes are installed.
% 1 if installed, 0 if not.
existToolbox = license('test','optimization_toolbox') && ...
    license('test','statistics_toolbox');
if ~existToolbox
    error('The ''CHAOSTEST'' function needs both Optimization and Statistics toolboxes.')
end
% Set initial conditions.
if (nargin >= 1) && (~isempty(X))

    if istable(X)              % Convert Table to an array.
        X = table2array(X);
    end
    if numel(X) == length(X)   % Check for a vector.
        X  =  X(:);             % Convert to a column vector.
    else
        error(' Observation ''Series'' must be a vector.');
    end
    % Remove any NaN (i.e. missing values) observation from 'Series'.

    X(isnan(X)) =   [];

else
    error(' Must enter at least observation vector ''Series''.');
end
%
% Specify the activation function to use in CHAOSFN and ensure it to be a
% string. Set default if necessary.
%
if nargin >= 2 && ~isempty(ActiveFN)
    if ~ischar(ActiveFN)
        error(' Activation function ''ActiveFN'' must be a string.')
    end
    % Specify the activation function to use:
    % ActiveFN = {'logistic', 'tanh', 'funfit'}.

    if ~any(strcmpi(ActiveFN , {'tanh' , 'logistic' , 'sigmoid'}))
        error('Activaton function ''ActiveFN'' must be TANH, LOGISTIC, or SIGMOID');
    end

else
    ActiveFN    =   'tanh';
end
%
% Ensure the maximum orders maxL, maxM and maxQ are positive integers, and
% set default if necessary.
%
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
% Check for a minimum size of vector X.
if length(X) <= maxL * maxM
    error(' Observation ''Series'' must have at least %d obeservations.',maxL*maxM+1)
end
%
% Ensure the significance level, ALPHA, is a
% scalar, and set default if necessary.
%
if (nargin >= 4) && ~isempty(alpha)
    if numel(alpha) > 1
        error(' Significance level ''Alpha'' must be a scalar.');
    end
    if (alpha <= 0 || alpha >= 1)
        error(' Significance level ''Alpha'' must be between 0 and 1.');
    end
else
    alpha  = 0.05;
end
%
% Check the method to carry out the test, by varying (L, m, q) or by fixing
% them, and set default if necessary. The method must be a string.
%
if nargin >= 5 && ~isempty(flag)
    if ~ischar(flag)
        error(' Regressions type ''FLAG'' must be a string.')
    end
else
    flag  =   'VARY';
end
if strcmpi(flag, 'FIX')
    StartL  =   maxL;
    StartM  =   maxM;
    StartQ  =   maxQ;
else
    StartL  =   1;
    StartM  =   1;
    StartQ  =   1;
end

if nargin < 6 || isempty(dumpfile), dumpfile = 'dump.mat'; end

% Initialize the Lyapunov Exponent to a small number.
Lambda =   - Inf;
Orders =   [maxL, maxM, maxQ];
% Initialized all value of Lambdas.
allLambda = -Inf(maxL, maxM, maxQ);
% Saving the values of the neural networks
ThetaAll   = cell(maxL, maxM, maxQ);
ResnormAll = NaN(maxL, maxM, maxQ);
%
% Create the structure OPTIONS to use for nonlinear least square function
% LSQNONLIN (included in the optimization toolbox). 
% 
% 
% Optimized for speed.
%
Options = optimoptions('lsqnonlin', ...
    'Display', 'off', ...
    'SpecifyObjectiveGradient', true, ...
    'Algorithm', 'levenberg-marquardt', ...
    'StepTolerance', 1e-8, ...
    'FunctionTolerance', 1e-8);

% Initialize the starting value for the coefficient THETA.
StartValue  =   0.5;
% Logging
nTotal = (maxL-StartL+1) * (maxM-StartM+1) * (maxQ-StartQ+1);
iter   = 0;
tStart = tic;
ExitflagAll = NaN(maxL, maxM, maxQ);
fprintf('chaostest_theta: %d fits, %d observations\n', nTotal, length(X));
for L = StartL:maxL
    for m = StartM:maxM
        for q = StartQ:maxQ
            % Starting with identical parameters kills the interest of
            % multpiple neurons
            % B = StartValue * ones(q, m); % Contains the coefficients beta(i,j).
            % C = StartValue * ones(q, 1); % Contains the coefficients beta(0,j).
            % A = StartValue * ones(1, q); % Contains the coefficients alpha(j).
            % D = StartValue; % alpha 0

            rng(42); % Reproducible symmetry breaking
            scale = 1 / std(X);
            B = randn(q, m) * scale * 0.5;
            C = randn(q, 1) * 0.5;
            A = randn(1, q) * (std(X) / sqrt(q));
            D = mean(X);

            Theta0 = [reshape(B', 1, q*m), C', A, D];

            [Theta,resnorm, ~, exitflag]  = lsqnonlin(@chaosfn, Theta0, [], [], Options, X, L, m, q, ActiveFN);
            ThetaAll{L,m,q}   = Theta;
            ResnormAll(L,m,q) = resnorm;
            ExitflagAll(L,m,q) = exitflag;

            T      =  length(X) - m*L; % New sample size.
            %
            % Use an inner function JACOBMATX to compute the Jacobian needed
            % to compute the Lyapunov Exponent LAMBDA.
            %
            JacobianMatrix   =   jacobmatx(Theta, X, L, m, q, ActiveFN);
            %
            % We distinguish between the "sample size" T used for estimating
            % the Jacobian, and the "block length" M which is the number of
            % evaluation points used for estimating LAMBDA (M <= T).
            %
            h   =   round(T ^ (1/3));
            newIndex    =   (h:h:T);
            M   =   length(newIndex);

            Tn = JacobianMatrix(:, :, newIndex);
            Q_qr = eye(m);
            r_diag_sum = 0;

            % Iterative QR to prevent matrix explosion (Eckmann-Ruelle)
            for t = 1:M
                [Q_qr, R_qr] = qr(Tn(:, :, t) * Q_qr);
                % Accumulate the log of the diagonal of R
                r_diag_sum = r_diag_sum + log(max(abs(R_qr(1, 1)), realmin));

                % Tn product for the Shintani-Linton variance estimator later
                if t > 1
                    Tn(:, :, t) = Tn(:, :, t) * Tn(:,:,t-1);
                end
            end

            % The dominant Lyapunov Exponent is the time-average of the first diagonal element
            Lambda1 = r_diag_sum / M;

            allLambda(L, m, q) = Lambda1; % Store it for future use.
            % Logging
            iter = iter + 1;
            el   = toc(tStart);
            fprintf('[%3d/%3d] (L,m,q)=(%d,%d,%d)  SSE=%.4g  exitflag=%d  lambda=%+.4f  elapsed %4.0fs  ETA %4.0fs\n', ...
                iter, nTotal, L, m, q, resnorm, exitflag, Lambda1, el, el/iter*(nTotal-iter));

            if (Lambda1 > Lambda)
                Change = true;
            elseif (Lambda1 == Lambda) && (L+m+q) < sum(Orders)
                Change = true;
            else
                Change = false;
            end

            if Change

                Lambda  = Lambda1;
                Orders  = [L, m, q];
                MBest = M;  hBest = h;  TBest = T;
                % Logging
                fprintf('          -> new best lambda = %+.4f at (%d,%d,%d)\n', Lambda, L, m, q);
                %
                % Compute the asymptotic variance of the Lyapunov Exponent for noisy
                % systems as computed by Shintani M. and Linton O. (2004).
                %
                Eta     =   zeros(M, 1);
                Zeta    =   Tn(:, :, 1)' * Tn(:, :, 1);
                Zeta    =   min(realmax, Zeta);
                Zeta    =   max(realmin, Zeta);

                Eta(1)  =   0.5 * log(max(realmin, max(eig(Zeta)))) - Lambda;
                for t = 2:M
                    Zeta1   = Tn(:, :, t)' * Tn(:, :, t);
                    Zeta1   = min(realmax, Zeta1);
                    Zeta1   = max(realmin, Zeta1);

                    Zeta2   = Tn(:, :, t-1)' * Tn(:, :, t-1);
                    Zeta2   = min(realmax, Zeta2);
                    Zeta2   = max(realmin, Zeta2);

                    Eta(t)  =   0.5 * log(max(realmin, max(eig(Zeta1)))./ ...
                        max(realmin, max(eig(Zeta2)))) - Lambda;
                end
                gamm    =   zeros(M, 1);
                for i = 1:M
                    gamm(i) =   1/M * Eta(M-i+1:M)' * Eta(1:i);
                end
                gamm   =   [gamm(1:end-1); flipud(gamm)];
                Sm      =   1.3221 * (1 * M)^(1/5);
                j           =   -M+1:M-1;
                KernelFN    =   ones(size(j));
                z           =   6 * pi * (j./Sm) / 5;
                z(j==0)     =   [];
                KernelFN(j~=0)    =   (3 ./ z.^2) .* (sin(z) ./ z - cos(z));
                varLambda       =   max(realmin, KernelFN * gamm);
                LambdaStatistic =   Lambda / sqrt(varLambda / M);

            end
        end
    end
end
%
pValue   =   normcdf(LambdaStatistic, 0, 1);
crit     = norminv(1 - alpha, 0, 1) * sqrt(varLambda / M);
CI       = [(Lambda - crit), Inf];
H  = (alpha >= pValue);

% Logging and saving dumpfile
fprintf('Done in %.0fs. Best (L,m,q)=(%d,%d,%d), lambda=%+.4f, p=%.3g\n', ...
    toc(tStart), Orders(1), Orders(2), Orders(3), Lambda, pValue);

% =========================================================================
% RECONSTRUCTION, PLOTTING, AND DIFFERENCE EVALUATION
% =========================================================================
bestL = Orders(1);
bestM = Orders(2);
bestQ = Orders(3);
bestTheta = ThetaAll{bestL, bestM, bestQ};

% Reconstruct fitted y over target sample
[residuals, ~] = chaosfn(bestTheta, X, bestL, bestM, bestQ, ActiveFN);
yTarget = X(bestM*bestL+1:end);
yReconstructed = yTarget - residuals';

% Calculate numerical differences
diffVector = yTarget - yReconstructed;
maxAbsDiff = max(abs(diffVector));
meanAbsDiff = mean(abs(diffVector));
rmseVal = sqrt(mean(diffVector.^2));
sseVal = sum(diffVector.^2);

fprintf('\n--- Reconstruction Differences ---\n');
fprintf('Sum of Squared Errors (SSE): %.6g\n', sseVal);
fprintf('Root Mean Squared Error (RMSE): %.6g\n', rmseVal);
fprintf('Mean Absolute Difference:    %.6g\n', meanAbsDiff);
fprintf('Maximum Absolute Difference: %.6g\n', maxAbsDiff);

% Plot original series vs reconstruction
figure('Name', 'Chaos Test: Time Series Reconstruction', 'NumberTitle', 'off');
timeIdx = (bestM*bestL+1):length(X);
plot(timeIdx, yTarget, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original Series');
hold on;
plot(timeIdx, yReconstructed, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Reconstructed Series');
grid on;
xlabel('Time Index (t)');
ylabel('Series Value');
title(sprintf('Original vs Reconstructed Series (Best L=%d, m=%d, q=%d)', bestL, bestM, bestQ));
legend('Location', 'best');
hold off;

save(dumpfile, 'H', 'pValue', 'Lambda', 'CI', 'Orders', 'allLambda', ...
    'ThetaAll', 'ResnormAll', 'ExitflagAll', ...
    'Eta', 'varLambda', 'LambdaStatistic', 'MBest', 'hBest', 'TBest', 'M', ...
    'alpha', 'ActiveFN', 'maxOrders', 'flag', 'StartValue', ...
    'yTarget', 'yReconstructed', 'rmseVal', 'maxAbsDiff');

%-------------------------------------------------------------------------%
%                       Helper function JACOBMATX                         %
%-------------------------------------------------------------------------%
function [y, J] = chaosfn(Theta, X, L, m, q, ActiveFN)
B   =   reshape(Theta(1:q*m), m, q)'; % size(B) = [q, m].
C   =   Theta(q*m+1:q*m+q)';          % size(C) = [q, 1].
A   =   Theta(q*m+q+1:q*m+q+q);       % size(A) = [1, q].
D   =   Theta(q*m+2*q+1);             % size(D) = 1.

XLAG   =  lagmatrix(X, L:L:L*m);   % size(XLAG) = [T, m].
XLAG   =  XLAG(m*L+1:end, :);      % Remove all NaN observation.
T      =  length(X) - m*L;         % New sample size.

% OLD: u = B * XLAG' + repmat(C,1,T);

% Faster implementation
% Implicit expansion removes the need to repmat C over T columns
u   =   B * XLAG' + C;             % size(u) = [q, T].
J   =   ones(length(Theta), T);

switch upper(ActiveFN)
    case 'LOGISTIC'
        y   =   D + A * (1 ./ (1 + exp(- u))); % size(y) = [1, T].

        % Implicit expansion for A removes repmats across T
        J(1:q*m,:)   =   repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(exp(- u) ./ (1 + exp(- u)).^2, m, 1)); % B.
        J(q*m+1:q*m+q,:) =  A' .* (exp(- u) ./ (1 + exp(- u)).^2); % C.
        J(q*m+q+1:q*m+q+q,:) =  1 ./ (1 + exp(- u)); % A.

    case 'TANH'
        y   =   D + A * tanh(u); % size(y) = [1, T].

        % Implicit expansion for A removes repmats across T
        J(1:q*m,:)   =   repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(sech(u).^2, m, 1)); % B.
        J(q*m+1:q*m+q,:) =  A' .* (sech(u).^2); % C.
        J(q*m+q+1:q*m+q+q,:) =  tanh(u); % A.

    case 'SIGMOID'
        y   =   D + A * (u .* (1 + abs(u / 2)) ./ (2 + abs(u) + u.^2 / 2));

        % Implicit expansion for A removes repmats across T
        J(1:q*m,:)   =   repmat(A', m, 1) .* (repmat(XLAG', q, 1) .* ...
            repmat(8*(1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2, m, 1)); % B.

        J(q*m+1:q*m+q,:) =  A' .* (8*(1 + abs(u)) ./ ...
            (3 + (1 + abs(u)).^2).^2); % C.
        J(q*m+q+1:q*m+q+q,:) =  u .* (1 + abs(u/2)) ./ (2 + abs(u) + u.^2 / 2); % A.

    otherwise
        error(' Unrecognized activation function!')
end
J   =   - J';
y   =   X(m*L+1:end)' - y;

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
    if L > 0 % Time delays
        YLag((L + 1):end,columns) = Y(1:(end - L), :);
    elseif L < 0 % Time leads
        YLag(1:(end + L),columns) = Y((1 - L):end, :);
    else % No shifts
        YLag(:,columns) = Y;
    end
end

%-------------------------------------------------------------------------%
function    J   =   jacobmatx(Theta, X, L, m, q, ActiveFN)
XLAG   =  lagmatrix(X, L:L:L*m); % size(XLAG) = [T, m].
XLAG   =  XLAG(m*L+1:end, :);    % Remove all NaN observation.
T      =  length(X) - m*L;       % New sample size.
B   =   reshape(Theta(1:q*m), m, q)'; % size(B) = [q, m].
C   =   Theta(q*m+1:q*m+q)';          % size(C) = [q, 1].
A   =   Theta(q*m+q+1:q*m+q+q);       % size(A) = [1, q].

% Implicit expansion eliminates repmat
u   =   B * XLAG' + C;    % size(u) = [q, T].
J   =   zeros(m, m, T);
J(2:end, 1:end-1, :)  =   repmat(eye(m-1), [1 1 T]);

switch upper(ActiveFN)
    case 'LOGISTIC'
        % Implicit expansion and reshape removes repmats across T
        J(1, :, :)  =   reshape(B' * (A' .* (exp(- u) ./ (1 + exp(- u)).^2)), 1, m, T);

    case 'TANH'
        % Implicit expansion and reshape removes repmats across T
        J(1, :, :)  =   reshape(B' * (A' .* (sech(u).^2)), 1, m, T);

    case 'SIGMOID'
        % Implicit expansion and reshape removes repmats across T
        J(1, :, :)  =   reshape(B' * (A' .* 8 .* (1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2), 1, m, T);

    otherwise
        error(' Unrecognized activation function!')
end