# Machine learning with Python

https://www.amazon.ca/Introduction-Machine-Learning-Python-Scientists/dp/1449369413

## Introduction

**Supervised learning**: automate decision-making processes by generalizing from known examples. Example: marking email as spam given a series of known spam emails.

**Unsupervised learning**: only the input data is known. No known output data is given to the algorithm. Example: segmenting customers into groups with similar preferences.

Representation of data is done through **tables**, where each row is a **sample** (e.g. a person), and each column is a **feature** (e.g. their age or their hair color).

`scikit-learn` is based on `NumPy` and `SciPi`. The fundamental data structure in `scikit-learn` is the `ndarray` class, an n-dimensional array.

http://www.scipy-lectures.org/

In order to measure the success of a machine learning model that uses supervised training we split the input data into two: the training set, and the test set. The `scikit-learn` function `train_test_split` does this, it shuffles data and splits 75% and 25%. After we trained the model, we can test its **accuracy** by running the model against the test set, and see how often it predicts the right target. 

**K-Nearest neighbor**: classification algorithm. Given a set of samples and a set of labels for each sample, it will predict the label for a new sample using the k nearest data points.