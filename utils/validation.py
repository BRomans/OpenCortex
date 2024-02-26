import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import numpy as np


def plot_feature_vector(x, x_flat, epoch=1):
    """
    Plot the feature vector and the original signal
    :param x: numpy array of shape (n_epochs, n_channels, n_samples)
    :param x_flat: numpy array of shape (n_epochs, n_channels * n_samples)
    :param epoch: int, index of the epoch to plot
    """
    plt.plot(x[epoch, 0], label='Channel 1', color='blue')
    offset = np.arange(200, 400)
    plt.plot(offset, x[epoch, 1], label='Channel 2', color='red')
    offset = np.arange(400, 600)
    plt.plot(offset, x[epoch, 2], label='Channel 3', color='orange')
    offset = np.arange(600, 800)
    plt.plot(offset, x[epoch, 3], label='Channel 4', color='purple')
    plt.plot(x_flat[epoch, :], label='Feature Vector', color='green', linestyle='dotted', linewidth=2)
    plt.legend(loc='upper right')
    plt.xlim(0, 800)
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


def plot_cross_validated_confusion_matrix(X, y, clf, cv, classes=None, normalize=False, title='Confusion Matrix',
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
    plt.show()
