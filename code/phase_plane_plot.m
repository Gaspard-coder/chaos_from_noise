function phase_plane_plot(x, dim, varargin)
% PHASE_PLANE_PLOT Generates a 2D or 3D phase-space trajectory.
%
% Syntax:
%   phase_plane_plot(x)                    % Defaults to 2D plot
%   phase_plane_plot(x, 2)                 % Explicit 2D (x vs dx)
%   phase_plane_plot(x, 3)                 % Explicit 3D (x vs dx vs d2x)
%   phase_plane_plot(x, 3, 'LineWidth', 1.5, 'Color', 'r')

% 1. Parse optional dimension argument
if nargin < 2 || ischar(dim) || isstring(dim)
    if nargin >= 2
        varargin = [{dim}, varargin]; % Shift option back to varargin
    end
    dim = 2; % Default to 2D trajectory
end

if ~ismember(dim, [2, 3])
    error('Dimension "dim" must be either 2 or 3.');
end

% 2. Enforce column vector & validate minimum data points
x = x(:);
if length(x) < dim
    error('Input vector x must contain at least %d data points.', dim);
end

% 3. Common plot styling options
plot_args = [{'Color', [0, 0.4470, 0.7410], 'LineWidth', 1}, varargin];

% 4. Compute derivatives and render plot based on dimension
if dim == 2
    % 2D: Signal vs Velocity
    y = diff(x);
    x_aligned = x(1:end-1);
    
    figure('Name', '2D Phase Space Representation', 'Color', 'w');
    plot(x_aligned, y, plot_args{:});
    
    xlabel('Signal x(t)', 'FontWeight', 'bold');
    ylabel('Velocity \Deltax(t)', 'FontWeight', 'bold');
    title('2D Phase Space Trajectory', 'FontWeight', 'bold');
    
else
    % 3D: Signal vs Velocity vs Acceleration
    y = diff(x);
    z = diff(y);
    x_aligned = x(1:end-2);
    y_aligned = y(1:end-1);
    
    figure('Name', '3D Phase Space Representation', 'Color', 'w');
    plot3(x_aligned, y_aligned, z, plot_args{:});
    
    xlabel('Signal x(t)', 'FontWeight', 'bold');
    ylabel('Velocity \Deltax(t)', 'FontWeight', 'bold');
    zlabel('Acceleration \Delta^{2}x(t)', 'FontWeight', 'bold');
    title('3D Phase Space Trajectory', 'FontWeight', 'bold');
    
    view(3);
    rotate3d on;
end

grid on;
axis tight;
end