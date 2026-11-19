% Normalizar la señal (media 0, desviación estándar 1) para estabilidad numérica
x = (x - mean(x)) / std(x);
N = length(x);

%% 2. ESTIMACIÓN AUTOMÁTICA DEL RETARDO (tau) POR AUTOCORRELACIÓN
% Buscamos el primer cruce por cero o caída por debajo de 1/e (0.3679)
max_lag = min(floor(N/10), 200);
acf = zeros(max_lag + 1, 1);
for lag = 0:max_lag
    acf(lag + 1) = mean(x(1:N-lag) .* x(1+lag:N));
end

tau = find(acf <= 1/exp(1), 1, 'first') - 1;
if isempty(tau) || tau < 1
    tau = 1; % Por defecto para mapas discretos
end

%% 3. ESTIMACIÓN AUTOMÁTICA DE LA DIMENSIÓN (m) POR FALSOS VECINOS (FNN)
m_max = 10;
R_tol = 15;     % Umbral estándar de Kennel et al.
A_tol = 2;
R_A = std(x);
fnn_porcentaje = zeros(m_max - 1, 1);

for m_test = 1:(m_max - 1)
    M_test = N - m_test * tau;
    if M_test < 50, break; end
    
    % Reconstrucción en dimensión m_test
    X_m = zeros(M_test, m_test);
    for j = 1:m_test
        X_m(:, j) = x((1:M_test) + (j - 1) * tau);
    end
    
    falsos = 0;
    % Muestreamos hasta 1000 puntos para acelerar el cálculo
    idx_muestra = round(linspace(1, M_test, min(M_test, 1000)));
    
    for idx = idx_muestra
        dist = sqrt(sum((X_m - X_m(idx, :)).^2, 2));
        dist(idx) = inf; % Excluir el mismo punto
        [d_m, vecino] = min(dist);
        
        if d_m > 0
            diff_nueva_coord = abs(x(idx + m_test * tau) - x(vecino + m_test * tau));
            d_m1 = sqrt(d_m^2 + diff_nueva_coord^2);
            
            % Criterios de falso vecino
            criterio1 = (diff_nueva_coord / d_m) > R_tol;
            criterio2 = (d_m1 / R_A) > A_tol;
            if criterio1 || criterio2
                falsos = falsos + 1;
            end
        end
    end
    fnn_porcentaje(m_test) = (falsos / length(idx_muestra)) * 100;
end

% Elegir m donde los falsos vecinos caen por debajo del 5%
m = find(fnn_porcentaje < 5, 1, 'first');
if isempty(m)
    [~, m] = min(fnn_porcentaje);
end
m = max(m, 2); % Dimensión mínima de 2

%% 4. VENTANA DE THEILER AUTOMÁTICA Y CÁLCULO DE DIVERGENCIA (ROSENSTEIN)
% La ventana de Theiler suele fijarse entre 2*tau y el período medio
ventana_theiler = max(10, 3 * tau);
max_pasos = max(30, 5 * tau);

M = N - (m - 1) * tau;
M_validos = M - max_pasos;

if M_validos <= ventana_theiler
    error('La serie es muy corta para los parámetros estimados (tau=%d, m=%d).', tau, m);
end

X = zeros(M, m);
for j = 1:m
    X(:, j) = x((1:M) + (j - 1) * tau);
end

divergencia = NaN(M_validos, max_pasos + 1);

for i = 1:M_validos
    distancias = sqrt(sum((X(1:M_validos, :) - X(i, :)).^2, 2));
    idx_excluir = max(1, i - ventana_theiler) : min(M_validos, i + ventana_theiler);
    distancias(idx_excluir) = inf;
    
    [d_min, idx_vecino] = min(distancias);
    
    if d_min > 0 && isfinite(d_min)
        for k = 0:max_pasos
            d_k = norm(X(i + k, :) - X(idx_vecino + k, :));
            if d_k > 0
                divergencia(i, k + 1) = log(d_k);
            end
        end
    end
end

y_k = mean(divergencia, 1, 'omitnan');
pasos_k = 0:max_pasos;

%% 5. AJUSTE LINEAL EN LA REGIÓN DE ESCALAMIENTO
% Se toma el primer tramo inicial donde ocurre la divergencia exponencial
k_inicio = 1;
k_fin = max(5, round(max_pasos * 0.35)); 

idx_lineal = (pasos_k >= k_inicio) & (pasos_k <= k_fin);
coeficientes = polyfit(pasos_k(idx_lineal), y_k(idx_lineal), 1);
lambda_muestra = coeficientes(1);

%% 6. RESULTADOS Y GRÁFICOS DE DIAGNÓSTICO
fprintf('--- PARÁMETROS ESTIMADOS AUTOMÁTICAMENTE ---\n');
fprintf('Retardo óptimo (tau): %d muestras\n', tau);
fprintf('Dimensión de incrustación (m): %d\n', m);
fprintf('Ventana de Theiler: %d muestras\n', ventana_theiler);
fprintf('--------------------------------------------\n');
fprintf('Máximo Exponente de Lyapunov: %.4f [1/muestra]\n', lambda_muestra);

figure('Color', 'w', 'Position', [100, 100, 1000, 400]);

% Subgráfico 1: Porcentaje de Falsos Vecinos
subplot(1, 2, 1);
plot(1:(m_max-1), fnn_porcentaje, 'ko-', 'LineWidth', 1.5); hold on;
%yline(5, 'r--', 'Umbral 5%');
plot(m, fnn_porcentaje(m), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
grid on;
xlabel('Dimensión de prueba (m)');
ylabel('Falsos Vecinos Más Cercanos (%)');
title(sprintf('Selección de Dimensión (m = %d)', m));

% Subgráfico 2: Curva de Divergencia y Pendiente
subplot(1, 2, 2);
plot(pasos_k, y_k, 'bo-', 'LineWidth', 1.2, 'MarkerSize', 4); hold on;
plot(pasos_k(idx_lineal), polyval(coeficientes, pasos_k(idx_lineal)), 'r--', 'LineWidth', 2.5);
grid on;
xlabel('Pasos discretos (k) [muestras]');
ylabel('Divergencia promedio \langle ln(d_k) \rangle');
title(sprintf('Exponente de Lyapunov: \\lambda = %.4f muestras^{-1}', lambda_muestra));
legend('Curva de divergencia', 'Ajuste lineal', 'Location', 'southeast');
