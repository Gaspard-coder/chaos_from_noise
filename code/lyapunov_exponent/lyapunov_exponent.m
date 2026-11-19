function [lp] = lyapunov_exponent(file,sample_frequency)

% =====
% This function allows to compute lyapunov exponent for a given
% .mat file acquired at the sample_frequency
% =====
% Input:
%     file : .mat file 
%     sample_frequency: frequency of acquisition of the studied signal
% 
% Output:
%     lp: lyapunov exponent
% =====

if nargin < 2 || isempty(sample_frequency)
    sample_frequency = 1;
end

if ischar(file) || isstring(file)
    loadedData = load(file);
    fields = fieldnames(loadedData);
    file = loadedData.(fields{1}); % Extracts the first variable stored in the .mat file
end

file = file(:)';

lp = lyapunovExponent(file,sample_frequency);