Keyboard Key Prediction using FFT & SVM
This project automates the process of detecting, segmenting, and classifying keyboard key presses (specifically Right Shift and Space) from audio recordings. It utilizes Digital Signal Processing (DSP) techniques for feature extraction and Support Vector Machines (SVM) for classification.
in one week
Project Creators: OUSSAMA AK-HAIL & ASSIA KARMOUT

🚀 Project Workflow
The project is divided into five main functional stages:

1. Automatic Segmentation
The script identifies individual key presses from a long audio recording.

Method: Peak detection based on a configurable amplitude threshold.

Parameters: Adjustable threshold, min_dist_seconds, and segment_duration.

2. Feature Extraction (FFT)
Converts time-domain audio signals into the frequency domain.

Processing: Converts audio to mono and applies Fast Fourier Transform (FFT).

Resampling: All frequency vectors are resampled to exactly 1000 features to ensure consistency.

3. Dataset & Data Formats
We provide the extracted features in multiple formats for flexibility:

dataset_fft.mat: Native MATLAB format for high-speed loading.

dataset_fft.xlsx: Excel format for manual inspection or use in other tools like Python/Tableau.

Labels: 0 for Right Shift, 1 for Space.

4. SVM Training & Visualization
Trains a Linear SVM to distinguish between the two keys.

3D PCA Projection: Since we have 1000 features, we use Principal Component Analysis to reduce them to 3 dimensions for visualization.

5. Predictor & Pre-trained Model
Model File (the_84.mat): A pre-trained SVM model with ~84% accuracy.

Samples: We have included sample .wav and .m4a files in the Samples/ folder to test the predictor immediately.
📈 Performance
The model currently achieves an accuracy of approximately 89.1%. It effectively distinguishes the low-frequency resonance of the spacebar from the higher-frequency mechanical click of the Shift key.

Note: When using the predictor, ensure the target_length remains at 1000 to match the model's training dimensions.
