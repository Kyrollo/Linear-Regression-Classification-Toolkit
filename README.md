# Linear Regression and Classification Toolkit

This repository contains Python functions for fitting a linear regression model to data, calculating the mean squared error, and classifying the results.

## Description

The project provides a set of Python functions designed to perform linear regression on a set of data points, calculate the mean squared error (MSSE), and classify the output. It also includes a function to add noise to the data to simulate real-world conditions and see how the model adapts.

## Features

- **lin_equation(w, x, c)**: Computes the linear equation \( y = wx + c \).
- **matrix_inverse(X)**: Computes the inverse of the matrix \( X \).
- **MSSE(X, Y)**: Computes the mean squared error and returns the weight vector \( W \).
- **classification(Y_hat)**: Classifies the output based on the sign of \( Y_{\text{hat}} \).
- **add_noise(Y)**: Adds Gaussian noise to the data.

## Usage

To use these functions, clone this repository to your local machine:

```sh
git clone https://github.com/<your-username>/<repository-name>.git
