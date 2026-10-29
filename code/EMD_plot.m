% =========================================================================
% EMD Decomposition & Multiresolution Visualization (Couleur Unifiée)
% =========================================================================

% 1. Extract Signal
x = frac_alea(1, :);
N = length(x);
t = 1:N; % Remplacez par votre vecteur temps si disponible (ex: t = (0:N-1)/fs)

% 2. Perform EMD Decomposition
[imf, residual, info] = emd(x, ...
    'SiftRelativeTolerance', 0.2, ...
    'SiftMaxIterations', 100, ...
    'MaxNumIMF', 10, ...
    'MaxNumExtrema', 1, ...
    'MaxEnergyRatio', 20, ...
    'Interpolation', 'spline');

% 3. Construct MRA Matrix and Partial Reconstruction
mra = [imf, residual]'; % Size: (numIMFs + 1) x N
numComponents = size(mra, 1);

% Ajustement automatique du masque de reconstruction
levelForReconstruction = [false, false, false, false, false, true, true, true, true, true, true];
if length(levelForReconstruction) < numComponents
    levelForReconstruction(end+1 : numComponents) = false;
else
    levelForReconstruction = levelForReconstruction(1 : numComponents);
end

reconstructedSignal = sum(mra(levelForReconstruction, :), 1);

% =========================================================================
% 4. Plot All Components in One Figure Window (Single Color)
% =========================================================================
numIMFs = size(imf, 2);
totalSubplots = numIMFs + 2; % Original/Reconstruction + IMFs + Residual

% Couleur unifiée (Bleu MATLAB standard)
plotColor = [0, 0.4470, 0.7410]; 

figure('Name', 'EMD Decomposition Components', 'Color', 'w', 'Position', [100, 50, 900, 800]);
tl = tiledlayout(totalSubplots, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = zeros(totalSubplots, 1);

% Panel 1: Signal Original vs Signal Reconstruit
ax(1) = nexttile;
plot(t, x, 'Color', plotColor, 'LineStyle', '-', 'LineWidth', 1.2, 'DisplayName', 'Original Signal');
hold on;
p=plot(t, reconstructedSignal, 'Color', plotColor, 'LineStyle', '-', 'LineWidth', 1.2, 'DisplayName', 'Reconstruction');
p.Color = "red";
ylabel('Signal');
title('EMD Multiresolution Analysis');
legend('Location', 'northeast');
grid on;

% Panels 2 to numIMFs + 1: IMFs individuelles
for k = 1:numIMFs
    ax(k+1) = nexttile;
    plot(t, imf(:, k), 'Color', plotColor, 'LineWidth', 1);
    ylabel(sprintf('IMF %d', k));
    grid on;
end

% Panel Last: Résidu
ax(end) = nexttile;
plot(t, residual, 'Color', plotColor, 'LineWidth', 1.2);
ylabel('Residual');
xlabel('Samples / Time');
grid on;

% Synchronisation des axes X pour le zoom/pan interactif
linkaxes(ax, 'x');