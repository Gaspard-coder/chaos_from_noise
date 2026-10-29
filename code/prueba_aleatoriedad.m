function prueba_aleatoriedad(x)
%close all
%clc
% Parámetros
fs = 1;%1000;             % Frecuencia de muestreo (Hz)
T = 10000;                 % Duración (s)
t = 0:1/fs:T-1/fs;     % Vector de tiempo
N = length(x);         % Número de muestras

% Señal aleatoria: Ruido gaussiano
%x = randn(1, N);       % Señal con distribución normal (media 0, varianza 1)

% ----------------------------
% 1. Estadísticas básicas
% ----------------------------
media = mean(x);
varianza = var(x);
desviacion = std(x);

% Mostrar resultados
fprintf('Media: %.4f\n', media);
fprintf('Varianza: %.4f\n', varianza);
fprintf('Desviación estándar: %.4f\n', desviacion);

% ----------------------------
% 2. Autocorrelación
% ----------------------------
[Rx, lags] = xcorr(x, 'biased'); % Autocorrelación normalizada

figure(1);
subplot(1,3,2)
plot(lags/fs, Rx);
xlabel('Lag (s)');
ylabel('Autocorrelation');
title('b)');
% ----------------------------
% 3. Densidad espectral de potencia (PSD)
% ----------------------------
[pxx, f] = pwelch(x, [], [], [], fs);  % Estimación con Welch

figure(1);
subplot(1,3,3)
plot(f, 10*log10(pxx));
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB/Hz)');
title('c)');
grid on;
% ----------------------------
% 4. Histograma (PDF estimada)
% ----------------------------
figure(1);
subplot(1,3,1)
histogram(x, 50, 'Normalization', 'pdf');
xlabel('Signal Value');
ylabel('Probability');
title('a)');
% ------------------------------
% 5. Reconstruccion E. de Fases
% ------------------------------
%  figure(1)
%  subplot(2,2,4)
%  xrec = x(1:length(x)-2);
%  yrec = diff(x(1:length(x)-1),1);
%  zrec = diff(x(1:length(x)),2)
%  plot3(xrec,yrec,zrec,'.b')
%  xlabel('xrec')
%  ylabel('yrec')
%  title('Reconstrucción de atractores sobre datos incognita 2')

