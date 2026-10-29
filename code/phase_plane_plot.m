function phase_plane_plot(x, varargin)
% PHASE_SPACE_3D Generates a 3D phase-space trajectory: x vs dx vs d2x.
%
% Syntax:
%   phase_space_3d(x)
%   phase_space_3d(x, 'LineWidth', 1.2)
%
% Inputs:
%   x        - 1D vector (time series data).
%   varargin - Optional plot parameters forwarded to plot3.

% 1. Enforce column vector
x = x(:);

if length(x) < 3
    error('Input vector x must contain at least 3 data points for 3D reconstruction.');
end

% 2. Calculate first and second derivatives
y = diff(x); % Length: N - 1
z = diff(y); % Length: N - 2

% 3. Align vector lengths to N - 2
x_aligned = x(1:end-2);
y_aligned = y(1:end-1); % Sliced to match length of z

% 4. Plot 3D Trajectory
figure('Name', '3D Phase Space Representation', 'Color', 'w');

% Default style: MATLAB standard blue line with varargin overrides
plot_args = [{'Color', [0, 0.4470, 0.7410], 'LineWidth', 1}, varargin];
plot3(x_aligned, y_aligned, z, plot_args{:});

% 5. Formatting
xlabel('Signal x(t)', 'FontWeight', 'bold');
ylabel('Velocity \Deltax(t)', 'FontWeight', 'bold');
zlabel('Acceleration \Delta^{2}x(t)', 'FontWeight', 'bold');
title('3D Phase Space Trajectory', 'FontWeight', 'bold');

grid on;
axis tight;
view(3); % Standard 3D view perspective
rotate3d on; % Enable interactive rotation
end