"""
This package contains functions to plot the feature vector, the data distribution, the confusion matrix, and the ROC curve.

Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2024 Michele Romani
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc

feat_colors = ['blue', 'red', 'orange', 'purple', 'gray', 'black', 'brown', 'pink',
               'green', 'cyan', 'magenta', 'yellow', 'olive', 'lime', 'teal', 'indigo', 'violet', 'salmon', 'gold']


def normalize(y, method='z-score'):
    if method == 'z-score':
        std_y = np.std(y)
        if std_y == 0:
            return np.zeros_like(y)
        return (y - np.mean(y)) / std_y
    elif method == 'softmax':
        return np.exp(y) / np.sum(np.exp(y))
    elif method == 'min-max':
        min_y = np.min(y)
        max_y = np.max(y)
        if max_y == min_y:
            return np.zeros_like(y)
        return (y - min_y) / (max_y - min_y)
    elif method == 'p-value':
        return 1 - y


def plot_feature_vector(x, x_flat, seg_len=200, epoch=1, xlim=(0, 1000)):
    """
    Plot the feature vector and the original signal
    :param seg_len: length of the segment
    :param x: numpy array of shape (n_epochs, n_channels, n_samples)
    :param x_flat: numpy array of shape (n_epochs, n_channels * n_samples)
    :param epoch: int, index of the epoch to plot
    """
    offset = np.arange(seg_len, seg_len)
    for i in range(x.shape[1]):
        offset = np.arange(i * seg_len, (i + 1) * seg_len)
        plt.plot(offset, x[epoch, i], label='Channel ' + str(i + 1), color=feat_colors[i])
    plt.plot(x_flat[epoch, :], label='Feature Vector', color='green', linestyle='dotted', linewidth=2)
    plt.legend(loc='upper right')
    plt.xlim(xlim)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.show()


def plot_data_distribution(x_train, x_test):
    """
    Plot the distribution of the mean of the feature vector
    :param x_train: numpy array of shape (n_epochs, n_channels, n_samples)
    :param x_test: numpy array of shape (n_epochs, n_channels, n_samples)
    """
    plt.hist(np.mean(x_train, axis=0), bins=30, alpha=0.5, label='Train', color='green')
    plt.hist(np.mean(x_test, axis=0), bins=30, alpha=0.3, label='Test', color='blue')
    plt.legend(loc='upper right')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of the mean of the feature vector')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix
    :param y_true: numpy array of shape (n_samples, )
    :param y_pred: numpy array of shape (n_samples, )
    :param classes: list of class names
    :param normalize: bool, whether to normalize the confusion matrix or not
    :param title: str, title of the plot
    :param cmap: matplotlib colormap
    """

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_cross_validated_confusion_matrix(X, y, clf, cv, classes=['Target', 'Non-Target'], normalize=False,
                                          title='Confusion Matrix',
                                          cmap=plt.cm.Blues):
    """ Plot the cross-validated confusion matrix
    :param X: numpy array of shape (n_samples, n_features)
    :param y: numpy array of shape (n_samples, )
    :param clf: classifier object
    :param cv: cross-validation method
    :param classes: list of class names
    :param normalize: bool, whether to normalize the confusion matrix or not
    :param title: str, title of the plot
    :param cmap: matplotlib colormap
    """
    # Perform cross-validated predictions
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block=False)


def plot_roc_curve(y_true, y_scores, pos_label=1, title='ROC Curve', figsize=(10, 6)):
    """
    Plots the ROC curve for a binary classification model.

    Parameters:
    - y_true: array-like of shape (n_samples,)
              True binary labels.
    - y_scores: array-like of shape (n_samples,)
                Target scores, can either be probability estimates of the positive class,
                confidence values, or non-thresholded measure of decisions.
    - title: str, optional
             Title of the plot.
    - figsize: tuple, optional
               Size of the figure.

    Returns:
    - None
    """
    # Compute the ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure(figsize=figsize)
    sns.set(style='whitegrid')

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_cross_validated_roc_curve(clf, X, y, pos_label=0, n_splits=10, random_state=None):
    """
    Plots a cross-validated ROC curve for a binary classification model.

    Parameters:
    - classifier: object
                  The classifier object with a 'fit' and 'predict_proba' method.
    - X: array-like of shape (n_samples, n_features)
         The input samples.
    - y: array-like of shape (n_samples,)
         The target values.
    - n_splits: int, default=5
                Number of folds for cross-validation.
    - figsize: tuple, optional
               Size of the figure.
    - random_state: int or RandomState instance, default=None
                    Controls the randomness of the cross-validation.

    Returns:
    - None
    """
    # Initialize the StratifiedKFold cross-validator
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize lists to store fpr, tpr, and auc for each fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Iterate over each fold
    for i, (train, test) in enumerate(cv.split(X, y)):
        # Fit the classifier on the training data
        clf.fit(X[train], y[train])
        # Predict probabilities on the test data
        y_pred = clf.predict(X[test])
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y[test], 1 - y_pred, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        # Interpolate the ROC curve at mean_fpr
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Start ROC curve at 0
        aucs.append(roc_auc)
        # Plot ROC curve for each fold
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {i + 1} (AUC = {roc_auc:.2f})')

    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

    # Plot random classifier
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross-Validated ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show(block=False)
