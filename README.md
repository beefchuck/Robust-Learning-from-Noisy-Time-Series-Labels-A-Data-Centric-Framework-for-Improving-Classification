# Robust Learning from Noisy Time Series Labels: A Data-Centric Framework for Improving Classification Reliability Across Application Domains
# Title

**Type:** Master Thesis

**Author:** Yannik Samuel Geiß

**1st Examiner:** Dr. Alona Zharova 

**2nd Examiner:** Prof. Dr. Stefan Lessmann


## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

Label noise — incorrect or unreliable training annotations—severely degrades supervised learning on time series data, where mislabeling errors propagate through temporal dependencies. Existing noise-robust methods assume independent samples and fail in sequential domains like medical monitoring and activity recognition, where annotation quality varies due to human error, sensor faults, and ambiguous temporal boundaries.

This thesis introduces TACL-Net (Temporal Attention-based Confidence Learning Network), integrating hierarchical convolutional encoding, temporal attention mechanisms, confidence-based label reliability estimation, and adaptive pseudo-labeling with sharpness-aware minimization. The architecture explicitly models temporal consistency to identify and correct corrupted labels during training without separate pretraining phases. Systematic evaluation on Human Activity Recognition and ECG5000 datasets at noise levels 0-60% demonstrates substantial advantages over baseline architectures. At 40% noise, TACL-Net achieves 82.42% accuracy versus baseline average of 71-75%. At extreme 60% noise on cardiac data, TACL-Net maintains 88.29% accuracy while baselines collapse to 58%—a 30-percentage-point improvement enabling deployment in uncontrolled clinical environments where clean training data is unattainable.

This repository provides the complete implementation and reproducible benchmarks for noise-robust time series classification.

**Keywords**: Noisy Labels, Time Series Classification, Label Noise Robustness, Robust Deep Learning, Label Reliability Estimation, Confidence Learning


## Working with the repo

### Dependencies

The code was written using Python 3.12 on MacOS. 
All application dependencies are included in the code -- the following dependencies/third libraries were used:

    - NumPy:^2.0.2
    - Pandas:^2.2.2
    - Matplotlib:^3.10.0
    - Scikit-Learn:^1.6.1
    - Seaborn:^0.13.2
    - PyTorch:^2.9.0+cu126

### Setup

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

1. Open the jupyter-notebooks in your favourite environment (e.g. Colab)
   
2. Upload the datasets for the corresponding models in the data-file of the juypter-notebook environment of your model
       a) ECG5000_TEST.txt and ECG5000_TRAIN.txt for ECG5000_TACL_and_Baseline.ipynb
       b) First unpack X_test.txt.zip and X_train.txt.zip and upload the unpacked files and y_test.txt and y_train.txt

3. Run the model

4. You are being asked which model you want to run, choose 1 to run the complete integrated experiment

5. When being asked to create comprehensive plots, type in "y"


## Results

The results folder contains the comprehensive figures.
