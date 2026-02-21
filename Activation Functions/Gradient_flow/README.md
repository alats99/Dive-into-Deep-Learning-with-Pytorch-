# Gradient Flow Investigation

This folder contains an investigation into the gradient flow of various activation functions in deep neural networks. 

## Overview

The goal of this investigation is to understand how different activation functions affect the magnitude and distribution of gradients as they propagate back through deep network layers. We specifically look at:
- Vanishing gradients (common in Sigmoid/Tanh networks)
- Exploding gradients (rare with proper initialization but possible)
- Dead neurons (common in ReLU networks)
- Stable gradient flow (aim of modern activations like Swish and ELU)

## Phillip Lippe's Tutorial Comparison

This work is inspired by **Phillip Lippe's Tutorial 3: Activation Functions and Gradient Flow** from the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/).

Citation:
```bibtex
@misc{lippe2024uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2024,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}
```

## Activation Functions Covered

- **Sigmoid**: Classic but prone to vanishing gradients due to small max gradient (0.25).
- **Tanh**: Similar to Sigmoid but zero-centered, with a higher max gradient (1.0).
- **ReLU**: Solves vanishing gradient for positive values but suffers from "dead neurons".
- **LeakyReLU**: Attemps to fix dead neurons by allowing a small negative slope.
- **ELU**: Similar to LeakyReLU but with a smooth exponential curve for negative values.
- **Swish**: Self-gated activation function ($x \cdot \sigma(x)$) often used in very deep networks.

## Files

- `activation_library.py`: Custom implementations of activation functions.
- `gradient_analysis.py`: Script to generate and visualize gradient flow across layers.

The analysis reveals how different initializations impact gradient propagation.

<!-- TABLE_DEFAULT -->
### Default Initialization Results

| Activation | Layer 1 (Log10) | Layer 10 (Log10) | Layer 20 (Log10) | Result |
| :--- | :--- | :--- | :--- | :--- |
| Sigmoid | 1.58e-20 (-19.8) | 1.45e-12 (-11.8) | 4.74e-04 (-3.3) | Vanished |
| Tanh | 1.77e-08 (-7.8) | 1.69e-08 (-7.8) | 1.75e-08 (-7.8) | Healthy |
| Relu | 2.17e-11 (-10.7) | 1.12e-11 (-10.9) | 1.14e-11 (-10.9) | Vanished |
| Leaky_relu | 1.94e-11 (-10.7) | 1.08e-11 (-11.0) | 7.33e-12 (-11.1) | Vanished |
| Elu | 1.59e-08 (-7.8) | 1.55e-08 (-7.8) | 1.61e-08 (-7.8) | Healthy |
| Swish | 2.37e-14 (-13.6) | 2.30e-14 (-13.6) | 2.40e-14 (-13.6) | Vanished |
| Gelu | 2.70e-14 (-13.6) | 2.59e-14 (-13.6) | 2.59e-14 (-13.6) | Vanished |
| Mish | 9.06e-13 (-12.0) | 8.74e-13 (-12.1) | 1.00e-12 (-12.0) | Vanished |
| Selu | 1.32e-05 (-4.9) | 1.24e-05 (-4.9) | 1.36e-05 (-4.9) | Healthy |
| Softplus | 4.46e-14 (-13.4) | 1.03e-08 (-8.0) | 2.12e-03 (-2.7) | Vanished |
| Hardswish | 2.40e-14 (-13.6) | 2.20e-14 (-13.7) | 2.31e-14 (-13.6) | Vanished |
| Sine | 1.71e-08 (-7.8) | 1.74e-08 (-7.8) | 1.76e-08 (-7.8) | Healthy |
<!-- TABLE_DEFAULT -->


<!-- TABLE_KAIMING -->
### Kaiming Initialization Results

| Activation | Layer 1 (Log10) | Layer 10 (Log10) | Layer 20 (Log10) | Result |
| :--- | :--- | :--- | :--- | :--- |
| Sigmoid | 2.48e-16 (-15.6) | 2.32e-10 (-9.6) | 4.61e-04 (-3.3) | Vanished |
| Tanh | 2.56e-03 (-2.6) | 9.66e-04 (-3.0) | 3.80e-04 (-3.4) | Healthy |
| Relu | 2.61e-03 (-2.6) | 5.22e-03 (-2.3) | 1.83e-03 (-2.7) | Healthy |
| Leaky_relu | 2.97e-03 (-2.5) | 4.56e-03 (-2.3) | 1.92e-03 (-2.7) | Healthy |
| Elu | 2.45e-02 (-1.6) | 3.48e-02 (-1.5) | 2.04e-02 (-1.7) | Healthy |
| Swish | 6.69e-06 (-5.2) | 3.05e-06 (-5.5) | 4.16e-06 (-5.4) | Healthy |
| Gelu | 2.00e-03 (-2.7) | 1.30e-03 (-2.9) | 2.36e-04 (-3.6) | Healthy |
| Mish | 1.30e-03 (-2.9) | 9.37e-04 (-3.0) | 2.98e-04 (-3.5) | Healthy |
| Selu | 2.39e-03 (-2.6) | 2.20e-03 (-2.7) | 1.53e-03 (-2.8) | Healthy |
| Softplus | 5.96e-09 (-8.2) | 9.08e-06 (-5.0) | 2.87e-03 (-2.5) | Healthy |
| Hardswish | 1.93e-06 (-5.7) | 1.26e-06 (-5.9) | 1.34e-06 (-5.9) | Healthy |
| Sine | 2.29e-04 (-3.6) | 2.28e-04 (-3.6) | 1.77e-04 (-3.8) | Healthy |
<!-- TABLE_KAIMING -->

<!-- TABLE_XAVIER -->
### Xavier Initialization Results

| Activation | Layer 1 (Log10) | Layer 10 (Log10) | Layer 20 (Log10) | Result |
| :--- | :--- | :--- | :--- | :--- |
| Sigmoid | 2.64e-16 (-15.6) | 2.32e-10 (-9.6) | 4.61e-04 (-3.3) | Vanished |
| Tanh | 2.01e-04 (-3.7) | 1.46e-04 (-3.8) | 1.31e-04 (-3.9) | Healthy |
| Relu | 7.82e-07 (-6.1) | 3.79e-07 (-6.4) | 2.60e-07 (-6.6) | Healthy |
| Leaky_relu | 6.67e-07 (-6.2) | 2.80e-07 (-6.6) | 2.40e-07 (-6.6) | Healthy |
| Elu | 1.43e-04 (-3.8) | 1.01e-04 (-4.0) | 9.53e-05 (-4.0) | Healthy |
| Swish | 9.72e-10 (-9.0) | 6.37e-10 (-9.2) | 6.87e-10 (-9.2) | Healthy |
| Gelu | 1.27e-09 (-8.9) | 8.55e-10 (-9.1) | 9.25e-10 (-9.0) | Healthy |
| Mish | 3.54e-08 (-7.5) | 2.49e-08 (-7.6) | 2.33e-08 (-7.6) | Healthy |
| Selu | 3.15e-03 (-2.5) | 2.17e-03 (-2.7) | 1.52e-03 (-2.8) | Healthy |
| Softplus | 5.61e-09 (-8.3) | 9.08e-06 (-5.0) | 2.87e-03 (-2.5) | Healthy |
| Hardswish | 8.88e-10 (-9.1) | 6.06e-10 (-9.2) | 6.19e-10 (-9.2) | Healthy |
| Sine | 2.74e-04 (-3.6) | 2.28e-04 (-3.6) | 1.74e-04 (-3.8) | Healthy |
<!-- TABLE_XAVIER -->

<!-- TABLE_ORTHOGONAL -->
### Orthogonal Initialization Results

| Activation | Layer 1 (Log10) | Layer 10 (Log10) | Layer 20 (Log10) | Result |
| :--- | :--- | :--- | :--- | :--- |
| Sigmoid | 2.47e-16 (-15.6) | 2.03e-10 (-9.7) | 4.58e-04 (-3.3) | Vanished |
| Tanh | 2.12e-04 (-3.7) | 1.24e-04 (-3.9) | 1.26e-04 (-3.9) | Healthy |
| Relu | 6.89e-07 (-6.2) | 2.14e-07 (-6.7) | 1.83e-07 (-6.7) | Healthy |
| Leaky_relu | 6.50e-07 (-6.2) | 2.21e-07 (-6.7) | 1.99e-07 (-6.7) | Healthy |
| Elu | 1.55e-04 (-3.8) | 9.25e-05 (-4.0) | 9.20e-05 (-4.0) | Healthy |
| Swish | 9.02e-10 (-9.0) | 5.42e-10 (-9.3) | 5.42e-10 (-9.3) | Healthy |
| Gelu | 1.15e-09 (-8.9) | 6.66e-10 (-9.2) | 6.67e-10 (-9.2) | Healthy |
| Mish | 3.51e-08 (-7.5) | 2.04e-08 (-7.7) | 2.05e-08 (-7.7) | Healthy |
| Selu | 2.68e-03 (-2.6) | 1.33e-03 (-2.9) | 1.14e-03 (-2.9) | Healthy |
| Softplus | 6.77e-09 (-8.2) | 9.09e-06 (-5.0) | 3.23e-03 (-2.5) | Healthy |
| Hardswish | 8.28e-10 (-9.1) | 5.06e-10 (-9.3) | 5.07e-10 (-9.3) | Healthy |
| Sine | 2.79e-04 (-3.6) | 1.65e-04 (-3.8) | 1.68e-04 (-3.8) | Healthy |
<!-- TABLE_ORTHOGONAL -->
