% =========================================================================
% DETERMINISM TEST FOR 1D SIGNALS (RECURRENCE PLOT)
% =========================================================================
%clear; clc; close all;

function DET = deter_test_2(x)
% -------------------------------------
N = length(x);
tipo_senal = 'To Study'
%% 2. Phase space recontruction parameters
m = 3;       % Embedding Dimension
tau = 2;     % Time Delay

% Reconstruction of the phase space
N_emb = N - (m - 1) * tau;
X_emb = zeros(N_emb, m);
for i = 1:m
    X_emb(:, i) = x((1 + (i-1)*tau):(N_emb + (i-1)*tau));
end

%% 3. Computing of the Distances Matrix and Recurrence Plot (RP)
% Computing the euclidean distance among all the reconstructed states 

dist_matrix = squareform(pdist(X_emb, 'euclidean'));

% Definition of the Threshold (Épsilon). Is frequent es el 10% of the max distance.
epsilon = 0.1 * max(dist_matrix(:));

% Binary Recurrence Matrix (1 if there ir, 0 if there isn't)
R = double(dist_matrix <= epsilon);

%% 4. Estimation of the Determinism Degree(RQA simplifyed)
% The determinism (DET) is computed as the percentage of the recurrence
% points that form a diagonal lines (excluding the main diagonal)
% This is a speedy version of the algorythm:

% Extracting the diagnonals (wihtout the main one)
lineas_diagonales = 0;
total_recurrencias = sum(R(:)) - N_emb; % substracting the main diag

for k = 1:(N_emb-2)
    diag_k = diag(R, k); % Sup Diagonal k
    % Counting of the neighboring points in the same diagonal.
    for j = 1:(length(diag_k)-1)
        if diag_k(j) == 1 && diag_k(j+1) == 1
            lineas_diagonales = lineas_diagonales + 2;
        end
    end
end
% Multiply for 2 to include simmetry (Inf. diag)
lineas_diagonales = lineas_diagonales * 2; 
DET = (lineas_diagonales / total_recurrencias) * 100;

%% 5. Visualización de Resultados
%figure('Position', [100, 100, 950, 450]);
figure(1)
% Subplot 1: La señal temporal
subplot(1, 2, 1);
plot(x, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
title(['Signal - ' tipo_senal]);
xlabel('Time (Samples)');
ylabel('Amplitude');
grid on;

% Subplot 2:  Recurrence plot
subplot(1, 2, 2);
imagesc(R);
colormap([1 1 1; 0 0 0]); % White for 0, black for 1 (Recurrence)
axis square;
title(sprintf('Recurrence Plot \\nDET estimado: %.2f%%', DET));
xlabel('State Index');
ylabel('State Index');
set(gca, 'YDir', 'normal');

fprintf('--- RESULT of Determinimism Test ---\n');
fprintf('Signal Type: %s\n', tipo_senal);
fprintf('Métric DET (Determinism): %.2f%%\n', DET);
if DET > 40
    fprintf('Conclusion: the signal has a main determinitic behaviour.\n');
else
    fprintf('Conclusion: the signal has a main random/noisy behaviour.\n');
end
