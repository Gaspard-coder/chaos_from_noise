% =========================================================================
% DETERMINISM TEST FOR 1D SIGNALS (RECURRENCE PLOT)
% Adapted to be used in the deter_test_batch function
% =========================================================================
%clear; clc; close all;
% Default parameters: m=3, tau=2, eps = 0.1

function DET = deter_test_3(x,m,tau,eps)
% -------------------------------------
N = length(x);
tipo_senal = 'To Study';
%% 2. Phase space recontruction parameters
%m = 3;       % Embedding Dimension
%tau = 2;     % Time Delay

% Reconstruction of the phase space
N_emb = N - (m - 1) * tau;
X_emb = zeros(N_emb, m);
for i = 1:m
    X_emb(:, i) = x((1 + (i-1)*tau):(N_emb + (i-1)*tau));
end

%% 3. Computing of the Distances Matrix and Recurrence Plot (RP)
% Computing the euclidean distance among all the reconstructed states

dist_matrix = squareform(pdist(X_emb, 'euclidean'));

% Definition of the Threshold (Épsilon). Is frequent es el eps of the max distance.
epsilon = eps * max(dist_matrix(:));

% Binary Recurrence Matrix (logical is faster and uses less memory than double)
R = dist_matrix <= epsilon;

%% 4. Estimation of the Determinism Degree (RQA simplified)
total_recurrencias = nnz(R) - N_emb; % subtracting the main diagonal

if total_recurrencias <= 0
    DET = 0; % No off-diagonal recurrences (prevents NaN)
else
    % Vectorized detection of diagonal lines of length >= 2
    % A point is in a valid diagonal if its up-left or down-right neighbor is 1.
    P = false(N_emb + 2);
    P(2:N_emb+1, 2:N_emb+1) = R;

    % Check neighbors using shifted matrices
    hasNeighbour = P(1:N_emb, 1:N_emb) | P(3:N_emb+2, 3:N_emb+2);

    % Mask with actual recurrences, looking only at the upper triangle
    inLine = triu(R & hasNeighbour, 1);

    % Multiply by 2 to include symmetric lower triangle
    DET = 100 * (2 * nnz(inLine)) / total_recurrencias;
end


%% 5. Visualización de Resultados
% %figure('Position', [100, 100, 950, 450]);
% figure(1)
% % Subplot 1: La señal temporal
% subplot(1, 2, 1);
% plot(x, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
% title(['Signal - ' tipo_senal]);
% xlabel('Time (Samples)');
% ylabel('Amplitude');
% grid on;
%
% % Subplot 2:  Recurrence plot
% subplot(1, 2, 2);
% imagesc(R);
% colormap([1 1 1; 0 0 0]); % White for 0, black for 1 (Recurrence)
% axis square;
% title(sprintf('Recurrence Plot \\nDET estimado: %.2f%%', DET));
% xlabel('State Index');
% ylabel('State Index');
% set(gca, 'YDir', 'normal');
%
% fprintf('--- RESULT of Determinimism Test ---\n');
% fprintf('Signal Type: %s\n', tipo_senal);
% fprintf('Métric DET (Determinism): %.2f%%\n', DET);
% if DET > 40
%     fprintf('Conclusion: the signal has a main determinitic behaviour.\n');
% else
%     fprintf('Conclusion: the signal has a main random/noisy behaviour.\n');
% end
