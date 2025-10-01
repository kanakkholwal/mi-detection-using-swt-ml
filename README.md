# Inferior Myocardial Infarction Detection using Stationary Wavelet Transform and Machine Learning

This project is a replication of the research paper:
**Inferior myocardial infarction detection using stationary wavelet transform and machine learning approach**
[Springer Link](https://link.springer.com/article/10.1007/s11760-017-1146-z)

## Overview

The goal of this project is to detect inferior myocardial infarction (IMI) from ECG signals using a combination of stationary wavelet transform (SWT) for feature extraction and machine learning algorithms for classification, as described in the referenced paper.

## Implementation Details

- **Dataset:**
  The PTB Diagnostic ECG Database is used, with patient data stored in the `ptbdb/` directory.

- **Preprocessing:**
  - ECG signals are loaded and segmented.
  - Noise and baseline wander are removed.
  - Data normalization is performed.

- **Feature Extraction:**
  - Stationary Wavelet Transform (SWT) is applied to ECG signals.
  - Relevant features are extracted from the wavelet coefficients.

- **Classification:**
  - Machine learning models (e.g., SVM, Random Forest) are trained on the extracted features.
  - Model performance is evaluated using accuracy, sensitivity, and specificity.

- **Scripts:**
  - `preprocess.py`: Handles data loading and preprocessing.
  - `replicate_swt_mi.py`: Main script for feature extraction and classification.
  - `debug.py`: Utility for debugging and analysis.

## How to Run

1. Install dependencies:

   ```python
   pip install -r requirements.txt
   ```

2. Run the main replication script:

   ```python
   python replicate_swt_mi.py
   ```

## Reference

If you use this code or results, please cite the original paper:
> Sharma, L., & Sunkaria, R. K. (2017). Inferior myocardial infarction detection using stationary wavelet transform and machine learning approach. Signal, Image and Video Processing, 12, 1043–1050. [https://doi.org/10.1007/s11760-017-1146-z](https://doi.org/10.1007/s11760-017-1146-z)
