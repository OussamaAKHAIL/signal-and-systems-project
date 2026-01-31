% This code is part of a Signals and Systems project with the goal of predicting the 
% pressed keys on a keyboard using the Fourier Transform and an SVM machine learning model
% made by : OUSSAMA AK-HAIL and ASSIA KARMOUT

%this part is for preparing your dataset after segmentation basically it do
%the fourier transform to each segment and save the first 1000 features

%% ========================================
%% AUTOMATIC FOLDER EXTRACTION
%% ========================================
clc;
clear all;
close all;

%% Configuration
target_length = 1000;  % first 1000 features 
base_path = 'C:\Users\akous\OneDrive\Desktop\potato\segmentation'; 

categories = {'right_shift', 'space'};
labels = [0, 1]; % 0 for right_shift, 1 for space

X_features = [];
y_labels = [];

%% Loop through each category folder
for c = 1:length(categories)
    current_category = categories{c};
    current_label = labels(c);
    
    % Path to the specific folder
    folder_path = fullfile(base_path, current_category);
    
    % Get list of all .wav files in that folder
    file_list = dir(fullfile(folder_path, '*.wav'));
    
    fprintf('Processing folder: %s (%d files found)\n', current_category, length(file_list));
    
    for i = 1:length(file_list)
        % Read the audio file
        file_name = fullfile(folder_path, file_list(i).name);
        [x, Fe] = audioread(file_name);
        
        % Pre-processing
        x = mean(x, 2); % Convert to mono
        Te = 1/Fe;
        N_segment = length(x);
        
        % Compute FFT
        % Frequency vector for the specific segment length
        f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
        X_segment = fftshift(fft(x) * Te);
        
        % Extract positive frequencies
        pos_idx = find(f_segment >= 0);
        fft_magnitude = abs(X_segment(pos_idx));
        
        % Normalize magnitude
        features = fft_magnitude' / (max(fft_magnitude) + eps);%eps is a safe switch to avoid deviding by zero
        
        
        % This ensures every file results in exactly 1000 features
        features_resampled = resample(features, target_length, length(features));
        
        % Add to dataset
        X_features = [X_features; features_resampled];
        y_labels = [y_labels; current_label];
    end
end

%% Final Summary
fprintf('\n--- Extraction Complete ---\n');
fprintf('Total samples: %d\n', size(X_features, 1));
fprintf('Feature vector length: %d\n', size(X_features, 2));
fprintf('Labels created: %d (0: right_shift, 1: space)\n', length(y_labels));
% Save the dataset for training
save('dataset_fft.mat', 'X_features', 'y_labels');