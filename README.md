# Elevating Rotating Machinery Fault Analysis: A Multifaceted Strategy with FFT, PCA, CNN and K-means

This repository contains the link data and code used in the article "Elevating Rotating Machinery Fault Analysis: A MultifacetedStrategy with FFT, PCA, CNN and K-means". 

## Introduction

This project works with the use of FFT and PCA to perform analyses of different types of gearbox operation based on the vibration signal.

## Dataset

The dataset is provided in the directory and consists of the following npy files:

- `x_1500_10.npy`: Contains the acquisition of accelerometers on the X axis.
- `y_1500_10.npy`: Contains the acquisition of accelerometers on the Y axis.
- `z_1500_10.npy`: Contains the acquisition of accelerometers on the Z axis.
- `gt_1500_10.npy`: Contains the labels.

## Notebook

The analysis is conducted in a python:

- `Códigos_TestePCA.py`: This code includes the model training, and evaluation of the predictive models using only PCA preprocessing. 
- `Códigos_TesteFFT.py`: This code includes the model training, and evaluation of the predictive models using only FFT preprocessing.
- `Códigos_TesteFFT_PCA.py`: This code includes the model training, and evaluation of the predictive models using FFT combinated with PCA preprocessing.
