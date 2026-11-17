function info = generate_labeled_signals(folder_path, N, n_realizations, seed)
% GENERATE_LABELED_SIGNALS Creates a labelled benchmark of 1D signals and
% saves each one as a .mat file (variable "x", column vector).
%
%   info = generate_labeled_signals(folder_path, N, n_realizations, seed)
%
%   folder_path     : output folder (created if needed)   [default 'labeled_signals']
%   N               : number of points per signal          [default 10000]
%   n_realizations  : independent realisations per type    [default 3]
%                     (parameters / initial conditions / noise are re-drawn
%                      for every realisation)
%   seed            : RNG seed for reproducibility         [default 42]
%
%   The label is encoded in the file name prefix:
%       DET_<type>_rXX.mat  -> deterministic signal
%       RND_<type>_rXX.mat  -> random / non-deterministic signal
%   deter_test_sweep reads the labels from this prefix.
%
%   DETERMINISTIC types
%     linear, quadratic, polynomial (degree 3-6), exponential, sinusoid,
%     two-tone (quasi-periodic), sawtooth, chirp,
%     logistic map (chaotic), Henon map (chaotic),
%     Lorenz (x component, chaotic), Rossler (x component, chaotic)
%
%   NON-DETERMINISTIC types
%     white Gaussian, white uniform, random binary (+-1),
%     AR(1) colored noise, smoothed (moving-average) noise, random walk
%     NOTE: the last three are "hard negatives": they are random but
%     temporally correlated/smooth, so RQA-DET tends to rate them high.
%
%   Output: table with FileName, Label, Type.
%
%   Example:
%       info = generate_labeled_signals('labeled_signals');

if nargin < 1 || isempty(folder_path),    folder_path = 'labeled_signals'; end
if nargin < 2 || isempty(N),              N = 10000; end
if nargin < 3 || isempty(n_realizations), n_realizations = 3; end
if nargin < 4 || isempty(seed),           seed = 42; end

if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end
rng(seed);

% {prefix, type name, generator handle}
gens = {
    'DET', 'linear',          @gen_linear
    'DET', 'quadratic',       @gen_quadratic
    'DET', 'polynomial',      @gen_polynomial
    'DET', 'exponential',     @gen_exponential
    'DET', 'sinusoid',        @gen_sinusoid
    'DET', 'two_tone',        @gen_two_tone
    'DET', 'sawtooth',        @gen_sawtooth
    'DET', 'chirp',           @gen_chirp
    'DET', 'logistic_map',    @gen_logistic
    'DET', 'henon_map',       @gen_henon
    'DET', 'lorenz_x',        @gen_lorenz
    'DET', 'rossler_x',       @gen_rossler
    'RND', 'white_gaussian',  @gen_white_gaussian
    'RND', 'white_uniform',   @gen_white_uniform
    'RND', 'random_binary',   @gen_random_binary
    'RND', 'ar1_colored',     @gen_ar1
    'RND', 'smoothed_noise',  @gen_smoothed_noise
    'RND', 'random_walk',     @gen_random_walk
    };

n_total   = size(gens, 1) * n_realizations;
FileName  = cell(n_total, 1);
Label     = cell(n_total, 1);
Type      = cell(n_total, 1);
row = 0;

for g = 1:size(gens, 1)
    for r = 1:n_realizations
        x = gens{g, 3}(N);
        x = x(:);
        if numel(x) ~= N || any(~isfinite(x))
            error('Generator "%s" produced an invalid signal (length or NaN/Inf).', gens{g, 2});
        end

        fname = sprintf('%s_%s_r%02d.mat', gens{g, 1}, gens{g, 2}, r);
        save(fullfile(folder_path, fname), 'x');

        row = row + 1;
        FileName{row} = fname;
        Type{row}     = gens{g, 2};
        if strcmp(gens{g, 1}, 'DET')
            Label{row} = 'Deterministic';
        else
            Label{row} = 'Random/Noisy';
        end
    end
end

info = table(FileName, Label, Type);
fprintf('Saved %d signals (%d points each) in "%s":\n', n_total, N, folder_path);
fprintf('  %d deterministic, %d random/noisy\n', ...
    sum(strcmp(Label, 'Deterministic')), sum(strcmp(Label, 'Random/Noisy')));
end

%% ========================================================================
%  DETERMINISTIC GENERATORS
%  ========================================================================
function x = gen_linear(N)
t = (0:N-1)' / (N-1);
a = (0.5 + rand) * sign(randn);
b = randn;
x = a * t + b;
end

function x = gen_quadratic(N)
t = 2 * (0:N-1)' / (N-1) - 1;
c = [(0.5 + rand) * sign(randn); randn; randn];
x = polyval(c, t);
end

function x = gen_polynomial(N)
t = 2 * (0:N-1)' / (N-1) - 1;
deg = randi([3 6]);
c = randn(deg + 1, 1);
c(1) = (0.5 + rand) * sign(randn);
x = polyval(c, t);
end

function x = gen_exponential(N)
t = (0:N-1)' / (N-1);
a = (1 + 3 * rand) * sign(randn);
x = exp(a * t);
end

function x = gen_sinusoid(N)
n = (0:N-1)';
P = 50 + 450 * rand;                 % period in samples
x = sin(2*pi*n / P + 2*pi*rand);
end

function x = gen_two_tone(N)
n  = (0:N-1)';
P1 = 50 + 150 * rand;
P2 = P1 * (sqrt(2) + 0.3 * rand);    % incommensurate -> quasi-periodic
x  = sin(2*pi*n / P1 + 2*pi*rand) ...
    + (0.5 + 0.5 * rand) * sin(2*pi*n / P2 + 2*pi*rand);
end

function x = gen_sawtooth(N)
n = (0:N-1)';
P = 50 + 450 * rand;
x = 2 * mod(n + P * rand, P) / P - 1;
end

function x = gen_chirp(N)
n  = (0:N-1)';
f0 = 1 / (100 + 200 * rand);
f1 = f0 * (2 + 2 * rand);
x  = sin(2*pi * (f0 * n + (f1 - f0) / (2*N) * n.^2));
end

function x = gen_logistic(N)
r = 3.7 + 0.29 * rand;               % chaotic regime (mostly)
s = 0.1 + 0.8 * rand;
nburn = 1000;
for k = 1:nburn
    s = r * s * (1 - s);
end
x = zeros(N, 1);
for k = 1:N
    s = r * s * (1 - s);
    x(k) = s;
end
end

function x = gen_henon(N)
a = 1.4; b = 0.3;
u = 0.1 * randn; v = 0.1 * randn;
nburn = 1000;
x = zeros(N, 1);
for k = 1:(N + nburn)
    u_new = 1 - a * u^2 + v;
    v = b * u;
    u = u_new;
    if k > nburn
        x(k - nburn) = u;
    end
end
end

function x = gen_lorenz(N)
f  = @(s) [10 * (s(2) - s(1)); s(1) * (28 - s(3)) - s(2); s(1) * s(2) - (8/3) * s(3)];
s0 = [1; 1; 1] + randn(3, 1);
x  = integrate_flow(f, s0, 0.01, 2, N, 500);       % sampling interval 0.02
end

function x = gen_rossler(N)
f  = @(s) [-s(2) - s(3); s(1) + 0.2 * s(2); 0.2 + s(3) * (s(1) - 5.7)];
s0 = randn(3, 1);
x  = integrate_flow(f, s0, 0.05, 2, N, 1000);      % sampling interval 0.1
end

function x = integrate_flow(f, s0, h, nsub, N, nburn)
% RK4 integration; returns the first state component, sampled every
% nsub steps, after discarding nburn samples of transient.
s = s0(:);
x = zeros(N, 1);
idx = 0;
total = (N + nburn) * nsub;
for k = 1:total
    k1 = f(s);
    k2 = f(s + 0.5 * h * k1);
    k3 = f(s + 0.5 * h * k2);
    k4 = f(s + h * k3);
    s  = s + (h / 6) * (k1 + 2*k2 + 2*k3 + k4);
    if mod(k, nsub) == 0
        idx = idx + 1;
        if idx > nburn
            x(idx - nburn) = s(1);
        end
    end
end
end

%% ========================================================================
%  NON-DETERMINISTIC GENERATORS
%  ========================================================================
function x = gen_white_gaussian(N)
x = randn(N, 1);
end

function x = gen_white_uniform(N)
x = rand(N, 1);
end

function x = gen_random_binary(N)
x = sign(randn(N, 1));
end

function x = gen_ar1(N)
phi = 0.7 + 0.25 * rand;
y = filter(1, [1 -phi], randn(N + 500, 1));
x = y(501:end);
end

function x = gen_smoothed_noise(N)
w = randi([5 20]);
y = filter(ones(w, 1) / w, 1, randn(N + w, 1));
x = y(w + 1:end);
end

function x = gen_random_walk(N)
x = cumsum(randn(N, 1));
end