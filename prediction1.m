% This code is part of a Signals and Systems project with the goal of predicting the 
% pressed keys on a keyboard using the Fourier Transform and an SVM machine learning model
% made by : OUSSAMA AK-HAIL and ASSIA KARMOUT

%this is the final part, here you can give a new sample signal and see how the model is predicting 


%% ========================================
%% PART 5: PREDICTOR 
%% ========================================
clc;
clear all;
close all;

%% 1. Configuration
% Path to your saved model
model_path = "C:\Users\akous\OneDrive\Documents\MATLAB\TP\New Folder\gp_model\the_84.mat";
%model_path = "C:\Users\akous\OneDrive\Desktop\MKA datasets\MKA datasets\hp\new Processed Data\epoch_svm.mat'

% test_audio = "C:\Users\akous\OneDrive\Desktop\MKA datasets\MKA datasets\zoom\Sound Segment(wav)\space\space6.wav"; 
% test_audio = "C:\Users\akous\OneDrive\Desktop\MKA datasets\MKA datasets\zoom\Sound Segment(wav)\Rshift\Rshift5.wav"; 
test_audio = "C:\Users\akous\Downloads\Rshift.m4a"; 

target_length = 1000; % Must match the training length !!!!!!!!

%% 2. Load the SVM model
if exist(model_path, 'file')
    load(model_path);
    fprintf('loaded successfully.\n');
else
    error('Model not found! Check your path.');
end

%% 3. Process the signal
% same traitement as in training 
[x, Fe] = audioread(test_audio);
x = mean(x, 2); % Convert to Mono


Te = 1/Fe;
N = length(x);
f = -Fe/2 : Fe/N : (Fe/2 - Fe/N);
X_fft = fftshift(fft(x) * Te);
pos_idx = find(f >= 0);
fft_magnitude = abs(X_fft(pos_idx));


features = fft_magnitude' / (max(fft_magnitude) + eps);


features_final = resample(features, target_length, length(features));

[prediction, score] = predict(svm_model, features_final); %here exactly where the prediction happens

%% 5.  Result
fprintf('\n=================================\n');
fprintf('       PREDICTION        \n');
fprintf('=================================\n');

if prediction == 1
    fprintf(' [ SPACE KEY ]\n');
else
    fprintf(' [ RSHIFT KEY ]\n');
end

% Confidence 
confidence = max(abs(score)); 
fprintf('Confidence :       %.4f\n', confidence);
fprintf('=================================\n');

% Optional: the plot of the mono fourier transform
figure;
plot(features_final);
title(['Feature Spectrum - Predicted: ' num2str(prediction)]);
xlabel('Frequency Bin'); ylabel('Normalized Magnitude');
grid on;