%% ========================================================
%% automatic segmentation 
%% ========================================================

% This code is part of a Signals and Systems project with the goal of predicting the 
% pressed keys on a keyboard using the Fourier Transform and an SVM machine learning model
% made by : OUSSAMA AK-HAIL and ASSIA KARMOUT

% This part of the code is for segmenting the data by detecting key presses through the peaks




clc; clear; close all;

%% 1. Configuration
input_file = "the file path"; 
output_folder = "the folder where too save the segments";
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

% avoiding over writing by offering unique name 
session_time = datestr(now, 'yyyymmdd_HHMMSS');

% Detection Parameters (based on observing the signal)
threshold = 0.020;          
min_dist_seconds = 0.3;    
segment_duration = 0.3;    
pre_trigger = 0.1;        

%% 2. Read and Process File
[x, Fe] = audioread(input_file);
x = mean(x, 2); 
t = (0:length(x)-1)/Fe;

%% 3. detecting the peaks 
abs_x = abs(x);
triggers = find(abs_x > threshold);
seg_matrix = [];
last_sample = 0;

for i = 1:length(triggers)
    current_sample = triggers(i);
    
    if current_sample > last_sample
        t_start = (current_sample / Fe) - pre_trigger;
        t_end = t_start + segment_duration;
        
        seg_matrix = [seg_matrix; t_start, t_end];
        last_sample = current_sample + (segment_duration * Fe);
        % seg_matrix = [seg_matrix; t_start, t_end];
        % last_sample = current_sample + (segment_duration * Fe);
    end
end

%% 4. plot and Export

figure('Name', 'Manual Trigger Plot', 'NumberTitle', 'off');
plot(t, x, 'Color', [0.5, 0.5, 0.5]); 
hold on; grid on;

for i = 1:size(seg_matrix, 1)
    t_start = seg_matrix(i, 1);
    t_end = seg_matrix(i, 2);
    
   
    filename = sprintf('Rshift_Auto_%s_%02d.wav', session_time, i);
    
    start_sample = max(1, round(t_start * Fe));
    end_sample = min(length(x), round(t_end * Fe));
    audiowrite(fullfile(output_folder, filename), x(start_sample:end_sample), Fe);
    
    % Plot boundaries
    line([t_start t_start], [-1 1], 'Color', 'g', 'LineStyle', '--');
    line([t_end t_end], [-1 1], 'Color', 'r', 'LineStyle', '--');
    patch([t_start t_end t_end t_start], [-1 -1 1 1], 'y', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end
title(['nbr de segments: ', num2str(size(seg_matrix, 1))]);


