# Machine learning with Python

https://www.amazon.ca/Introduction-Machine-Learning-Python-Scientists/dp/1449369413

## Introduction

**Supervised learning**: automate decision-making processes by generalizing from known examples. Example: marking email as spam given a series of known spam emails.

**Unsupervised learning**: only the input data is known. No known output data is given to the algorithm. Example: segmenting customers into groups with similar preferences.

Representation of data is done through **tables**, where each row is a **sample** (e.g. a person), and each column is a **feature** (e.g. their age or their hair color).

`scikit-learn` is based on `NumPy` and `SciPy`. The fundamental data structure in `scikit-learn` is the `ndarray` class, an n-dimensional array. See http://www.scipy-lectures.org.

In order to measure the success of a machine learning model that uses supervised training we split the input data into two: the training set, and the test set. The `scikit-learn` function `train_test_split` does this, it shuffles data and splits 75% and 25%. After we trained the model, we can test its **accuracy** by running the model against the test set, and see how often it predicts the right target.

If a model can make accurate predictions on unseen data, it is able to **generalize** from the training set to the test set. We want to build models that can generalize as accurate as possible. However, sometimes models are built too complex, and so they are **overfitted** models. This usually happens when we have too small of a training set, and the model is built off rules that are very specific to this set, so it cannot generalize very well. If the model is very simple, it is an **underfitted** model. A less complex model means worse performance on the training set, but better generalization.

## Supervised learning

There are 2 types of supervised machine learning problems:

- Classification: the goal is to predict the category of an object. E.g. is an email spam or not?

- Regression: the goal is to predict a continuous number. E.g. what is the predicted income of a person given their age and gender?

**K-Nearest neighbor**: classification algorithm. Given a set of samples and a set of labels for each sample, it will predict the label for a new sample using the `k` nearest data points. When `k > 1`, we count how many neighbors belong to each class and then the class with the higher number of "votes" wins. Using few neighbors leads to high model complexity, and using many neighbors corresponds to low model complexity. There is also a regression variant of this algorithm. It is not often used in practice because it is slow and cannot handle many features.

**Linear models**: they make a prediction using a linear combination of the input features. `y = b + w0 x0 + w1 x1 + ... + wN xN` . The prediction is a line for a single feature, a plane for two features, and hyperplanes when 3 or more features exist on the input data. The difference between all the models is how the model parameters `w` and `b` are learned from the training data. `w` is the slope(s) and `b` is the intercept parameter. They produce models for regression and for classification. The models for binary classification are such that if the prediction is less than zero, we predict the class -1, and if it is larger than zero, we predict the class +1. For regression problems we just return the weighted sum of the features.

For regression problems:

- Linear regression (a.k.a ordinary least squares): a regression algorithm. It finds the parameters `w` and `b` such that it minimizes the mean squared error between predictions and true regression targets. It has no parameters, but it also has no way to control model complexity.

- Ridge regression: a regression algorithm. The coefficients `w` are chosen not only so that they predict well on the training data, but also so that they are as small as possible. Each feature should have as little effect on the outcome as possible. This is called **regularization**. This model receives a parameter `alpha`. When `alpha` is high the coefficients become closer to zero, which decreases training set performance but might help with generalization.

- Lasso regression: a regression algorithm similar to Ridge regression, but the coefficients `w` are allowed to be exactly zero. This means that it may completely ignore features of the dataset. It also accepts an `alpha` parameter. Lower `alpha`s build more complex models.

For classification problems:

- Logistic regression: a **classification** algorithm despite its name. It accepts a parameter `C`. When `C` is high this model tries to fit the training set as best as possible.

- Linear support vector machines (linear SVMs). It accepts a parameter `C`. When `C` is high this model tries to fit the training set as best as possible.

- Naive Bayes Classifiers. They learn parameters by looking at each feature individually and then collecting simple per-class statistics from each feature. To make a prediction, a data point is compared to the statistics of each class, and the best matching class is predicted.

    - Gaussan NB: can be applied to continuous data
    - Bernoulli NB: assumes binary data
    - Multinomial NB: assumes count data

Q: why is it said on the book that "as training and test set performance are very close, it is likely that we are underfitting"?

It is possible to build linear models that apply to multiclass problems, i.e. it can predict the category of an object when there are more than 2 categories. One technique is the "one-vs.-rest" approach. Build binary models for each class, and then run them all on the test point. The classifier with the highest score "wins".

### Decision Trees

They can be used for both classification and regression tasks. They learn a hierarchy of if/else questions (called "tests") that lead to a decision. The tests that are used on continuous data are of the form "is feature `i` larger than value `a`?" This builds a partition of the data set. The recursive partitioning of the data can continue until each region in the partition (each leaf in the decision tree) only contains a single class. If this happens the leafs of the decision tree are called "pure".

A prediction on a new data point is made by checking which region of the partition of the feature space the point lies in.

If the decision tree only has pure leaves it can be a signal for an overfitted model that will have 100% accuracy on the training set but not very good accuracy on the test set.

Some techniques to control the complexity of the decision tree: collapsing nodes that contain little information, limiting the maximum depth of the tree, limiting the maximum number of leaves.

One thing to note: tree-based **regression** models don't extrapolate, i.e. they cannot make predictions outside of the range of the training data.

The main disadvantage of decision trees is that they tend to overfit. Ensembles of decision trees are usually used in place of a single decision tree.

- Random forests: collection of decision trees, each of which is built differently: by selecting the data points, and by selecting the features in each split test. Their creation can be parallelized.
- Gradient boosted decision trees: each tree tries to correct the mistakes of the one built previously. The main parameters of gradient boosted tree models are the number of trees, `n_estimators`, and the `learning_rate` which controls the degree to which each tree is allowed to correct the mistakes of the previous tree. A lower `learning_rate` means that more trees are needed to build a model of similar complexity.