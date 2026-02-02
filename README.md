# Max Healthcare AI/ML Assignment

This repository contains my submission for the Max Healthcare take-home assessment conducted via the Training & Placement Cell, NSUT Delhi.

## Problem Overview
The task involves building a robust classification model for dermatology images where training labels may be noisy, while validation data is clean and expert-verified.

## Approach
- Performed exploratory data analysis to understand class distribution and visual patterns.
- Designed a lightweight CNN architecture suitable for 28×28 grayscale medical images.
- Applied regularization and label smoothing to improve robustness against noisy labels.
- Selected the best model based on validation accuracy.

## Repository Structure
- `notebook/`: Main Jupyter notebook with EDA, training, and evaluation.
- `model/`: Saved best-performing model weights.
- `inference/`: Script for loading a new dataset and evaluating accuracy.

## Live Inference
A reusable inference function is provided to evaluate hidden test datasets during on-campus interviews.

## Author
Vikash Singhaniya  
M.Tech, NSUT Delhi
