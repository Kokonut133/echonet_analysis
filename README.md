# EchoNext ECG Structural Heart Disease Prediction

This project explores the EchoNext PhysioNet dataset for ECG-based detection of echocardiogram-confirmed structural heart disease.

The dataset contains 100,000 de-identified 12-lead ECG recordings paired with structural heart disease labels derived from echocardiography. Each ECG is sampled at 250 Hz and includes demographic and ECG metadata such as age, sex, heart rate, PR interval, QRS duration, and corrected QT interval.

## Goal

The goal of this project is to build a transparent, clinically oriented machine learning pipeline for predicting structural heart disease from ECG data.

The project focuses on:

* exploratory analysis of 12-lead ECG waveform data
* comparison of metadata-only and waveform-based models
* clinically meaningful validation metrics
* interpretable model behavior
* clear visualizations suitable for technical and clinical audiences

## Planned Workflow

### 1. Data Exploration

I will first inspect the dataset structure, metadata completeness, label distribution, and waveform characteristics. This includes visualizing individual 12-lead ECGs, comparing positive and negative structural heart disease cases, and analyzing demographic and ECG feature distributions.

### 2. Baseline Models

The first modeling step will use tabular ECG metadata only, such as age, sex, heart rate, PR interval, QRS duration, and QTc. This provides a simple clinical baseline before adding raw waveform information.

### 3. Waveform Feature Engineering

Next, I will extract lead-wise summary features from the ECG waveforms, including amplitude statistics, signal energy, frequency-domain features, and morphology-related proxies. These features will be compared against the metadata-only baseline.

### 4. Deep Learning on Raw ECGs

After establishing baselines, I will train a lightweight 1D convolutional neural network on raw 12-lead ECG waveforms to predict structural heart disease. The model will be evaluated using AUROC, AUPRC, sensitivity/specificity tradeoffs, and subgroup performance.

### 5. Error Analysis and Interpretability

The final analysis will inspect false positives and false negatives, evaluate performance across demographic and ECG subgroups, and visualize which leads or signal regions contribute most strongly to predictions.

## Why This Project

This project demonstrates an end-to-end healthcare AI workflow: physiological signal processing, clinical label prediction, model validation, visualization, and interpretation. The focus is not only on model performance, but on building a reproducible and explainable pipeline for medical machine learning.
