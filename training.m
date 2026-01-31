
% This code is part of a Signals and Systems project with the goal of predicting the 
% pressed keys on a keyboard using the Fourier Transform and an SVM machine learning model
% made by : OUSSAMA AK-HAIL and ASSIA KARMOUT

%this part is for training and saving the SVM model



%% ========================================
%% TRAIN & SAVE SVM 
%% ========================================
clc;
clear all;
close all;

%% 1. Configuration
data_folder ="C:\Users\akous\OneDrive\Documents\MATLAB\TP\New Folder\gp_model";
dataset_file = fullfile(data_folder, 'dataset_fft.mat');
% data_folder ="C:\Users\OneDrive\Documents\MATLAB\TP\New Folder\model";
% dataset_file = fullfile(data_folder, 'dataset_fft.mat');

%% 2. Load Data
fprintf('Loading dataset from: %s\n', dataset_file);
if exist(dataset_file, 'file')
    load(dataset_file); 
else
    error('File not found! Check your path.');
end
% else
%     error('File not found! Check your path.');
% end

fprintf('Loaded %d samples.\n', length(y_labels));

%% 3. Shuffle & Split Data 

n = length(y_labels);
p = randperm(n);
X = X_features(p, :);
y = y_labels(p, :);


cv = cvpartition(y, 'HoldOut', 0.3);% ---> 70% fpr training and 30% for testing
idx_train = training(cv);
idx_test = test(cv);

X_train = X(idx_train, :);
y_train = y(idx_train, :);
X_test = X(idx_test, :);
y_test = y(idx_test, :);

fprintf('Training set: %d samples\n', length(y_train));
fprintf('Testing set:  %d samples\n', length(y_test));

%% 4. Train the SVM
fprintf('Training SVM model... ');

% 'Standardize' is important to normalize the low and high energie frequancies
svm_model = fitcsvm(X_train, y_train, ...
    'KernelFunction', 'linear', ...
    'Standardize', true, ...
    'ClassNames', [0, 1]); 
fprintf('Done!\n');

%% 5. Evaluate Accuracy
[predictions, scores] = predict(svm_model, X_test);
accuracy = sum(predictions == y_test) / length(y_test) * 100;

fprintf('\n---------------------------------\n');
fprintf('Model Accuracy: %.2f%%\n', accuracy);
fprintf('---------------------------------\n');

% Confusion Matrix (Visual Proof)
figure('Name', 'SVM Performance');
confusionchart(y_test, predictions);
title(sprintf('SVM Accuracy: %.1f%%', accuracy));

%% 6. Save the Trained Model

model_filename = fullfile(data_folder, 'the_84.mat');
save(model_filename, 'svm_model');

fprintf('Trained model saved to:\n%s\n', model_filename);

