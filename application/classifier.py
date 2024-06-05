import logging
import numpy as np
from brainflow import BoardShim, BoardIds
from imblearn.over_sampling import RandomOverSampler
from mne import set_eeg_reference, find_events, Epochs
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from utils.layouts import layouts
from utils.loader import convert_to_mne
from utils.preprocessing import basic_preprocessing_pipeline, extract_epochs
from utils.validation import plot_cross_validated_roc_curve, plot_cross_validated_confusion_matrix

random_state = 32
np.random.seed(random_state)

models = {
    'SVM': SVC(kernel='linear', C=1, random_state=random_state),
    'LDA': LinearDiscriminantAnalysis(),
}


class Classifier:
    """
    Classifier class to train and evaluate custom models on EEG data
    """

    def __init__(self, model, board_id):
        if model is None:
            raise ValueError("Model cannot be None")
        self.model = models[model]
        self.board_id = board_id
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.chs = layouts[self.board_id]["channels"]
        self.epoch_start = -0.2
        self.epoch_end = 0.5
        self.seg_start = 50
        self.seg_end = 150
        self.baseline = (self.epoch_start, 0)
        self.train_X = None
        self.train_y = None
        self.prep_X = None
        self.prep_Y = None

    def preprocess(self, data):
        """
        Preprocess the data and extract epochs
        :param data: numpy array of shape (n_samples, n_channels)
        """
        start_eeg = layouts[self.board_id]["eeg_start"]
        end_eeg = layouts[self.board_id]["eeg_end"]
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1]
        raw = convert_to_mne(eeg=eeg, trigger=trigger, rescale=1e6, fs=self.fs, chs=self.chs, recompute=False,
                             transpose=False)
        events = find_events(raw, stim_channel='STI', initial_event=True, shortest_event=1)
        events[:, 2][events[:, 2] != 1] = 3
        if BoardIds.ENOPHONE_BOARD == self.board_id:
            raw_data, _ = set_eeg_reference(raw, ref_channels='average')
        filtered = basic_preprocessing_pipeline(raw, lp_freq=2, hp_freq=15, notch_freqs=(50, 60))
        eps = extract_epochs(data=filtered, events=events, tmin=self.epoch_start, tmax=self.epoch_end,
                             baseline=self.baseline)

        self.train_X = eps.get_data()[:, :, self.seg_start:self.seg_end]
        self.train_y = eps.events[:, -1]
        logging.info(f"Data preprocessed and epochs extracted with shape {self.train_X.shape}")

    def train(self, data, scaler=StandardScaler(), oversample=False, random_state=random_state):
        """
        Train the model on the preprocessed data
        :param data: numpy array of shape (n_samples, n_channels)
        :param scaler: Scaler object
        :param oversample: bool, whether to oversample the minority class or not
        :param random_state: int, random state for reproducibility
        """
        self.preprocess(data)
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=random_state)
        le = LabelEncoder()
        X = self.train_X.reshape(self.train_X.shape[0], -1)
        y = le.fit_transform(self.train_y)

        if oversample:
            X, y = oversampler.fit_resample(X, y)

        self.prep_X = X
        self.prep_Y = y

        X = scaler.fit_transform(X)

        self.cross_validate(X, y)

        self.model.fit(X, y)

    def cross_validate(self, X, y, n_splits=5):
        """
        Cross-validate the model
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples, )
        :param n_splits: int, number of splits for cross-validation
        :return: cross-validated accuracy and F1 scores
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_accuracy = cross_val_score(self.model, X, y, cv=cv)
        cv_f1 = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        logging.info(f"Cross-validation accuracy: {cv_accuracy.mean():.2f} +/- {cv_accuracy.std():.2f}")
        logging.info(f"Cross-validation F1: {cv_f1.mean():.2f} +/- {cv_f1.std():.2f}")
        return cv_accuracy, cv_f1

    def predict_class(self, data):
        """
        Compute the class labels for the input data
        :param data: numpy array of shape (n_samples, n_features)
        :return: numpy array of shape (n_samples, )
        """
        return self.model.predict(data)

    def predict_proba(self, data):
        """
        Compute the probability estimates for the input data
        :param data: numpy array of shape (n_samples, n_features)
        :return: numpy array of shape (n_samples, n_classes)
        """
        return self.model.predict_proba(data)

    def plot_roc_curve(self, n_splits=5):
        """ Plot the cross-validated ROC curve for the model"""
        plt.close()
        plot_cross_validated_roc_curve(clf=self.model, X=self.prep_X, y=self.prep_Y, pos_label=0, n_splits=n_splits,
                                       random_state=random_state)

    def plot_confusion_matrix(self, n_splits=5):
        """ Plot the cross-validated confusion matrix for the model"""
        plt.close()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        plot_cross_validated_confusion_matrix(X=self.prep_X, y=self.prep_Y, clf=self.model, cv=cv)
