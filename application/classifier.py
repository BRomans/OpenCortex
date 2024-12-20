import logging
import mne.utils
import numpy as np
import matplotlib
from brainflow import BoardShim, BoardIds
from imblearn.over_sampling import RandomOverSampler
from mne import set_eeg_reference, find_events, Epochs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from utils.layouts import layouts
from utils.loader import convert_to_mne
from processing.preprocessing import make_overlapping_epochs
from validation.plotting import plot_cross_validated_roc_curve, plot_cross_validated_confusion_matrix, normalize

# turn off MNE logging
mne.utils.set_log_level('ERROR')
matplotlib.use("Qt5Agg")
random_state = 32
np.random.seed(random_state)

models = {
    'SVM': SVC(kernel='linear', C=1, probability=True, random_state=random_state),
    'LDA': LinearDiscriminantAnalysis(),
    'RF': RandomForestClassifier(n_estimators=10, random_state=random_state)
}


class Classifier:
    """
    Classifier class to train and evaluate custom models on EEG data
    """

    def __init__(self, model, board_id):
        if model is None:
            raise ValueError("Model cannot be None")
        self.model = models[model]
        self.mode = 'train'
        self.sequence = []
        self.board_id = board_id
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.chs = layouts[self.board_id]["channels"]
        self.epoch_start = -0.1
        self.epoch_end = 0.7
        self.cv_splits = 5
        self.seg_start = 50
        self.seg_end = 150
        self.baseline = (self.epoch_start, 0)
        self.scaler = None
        self.train_X = None
        self.train_y = None
        self.prep_X = None
        self.prep_Y = None
        self.eval_X = None
        self.eval_y = None

    def train(self, data, scaler=StandardScaler(), oversample=False, random_state=random_state):
        """
        Train the model on the preprocessed data
        :param data: numpy array of shape (n_samples, n_channels)
        :param scaler: Scaler object
        :param oversample: bool, whether to oversample the minority class or not
        :param random_state: int, random state for reproducibility
        """
        prep_data, labels = self.preprocess(data)

        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=random_state)
        le = LabelEncoder()
        X = prep_data.reshape(prep_data.shape[0], -1)
        y = le.fit_transform(labels)
        logging.info(f"Encoded labels: {y}")

        self.prep_X = X
        self.prep_Y = y

        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=random_state)

        if oversample:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)

        X_train = scaler.fit_transform(X_train)
        self.scaler = scaler
        self.cross_validate(X_train, y_train, n_splits=self.cv_splits)
        self.model.fit(X_train, y_train)
        self.evaluate(X_eval, y_eval)

    def cross_validate(self, X, y, n_splits=5):
        """
        Cross-validate the model
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples, )
        :param n_splits: int, number of splits for cross-validation
        :return: cross-validated accuracy and F1 scores
        """
        method = 'predict'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_accuracy = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        f1_scorer = make_scorer(f1_score, average='weighted')
        cv_f1 = cross_val_score(self.model, X, y, cv=cv, scoring=f1_scorer)
        logging.info(f"Cross-validation accuracy: {cv_accuracy.mean():.2f} +/- {cv_accuracy.std():.2f}")
        logging.info(f"Cross-validation F1: {cv_f1.mean():.2f} +/- {cv_f1.std():.2f}")
        return cv_accuracy, cv_f1

    def evaluate(self, X, y):
        """
        Evaluate the model on the test data
        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples, )
        :return: accuracy and F1 scores
        """
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        f1 = f1_score(y, y_pred, average='weighted')
        logging.info(f"Eval Accuracy: {accuracy:.2f}")
        logging.info(f"Eval F1: {f1:.2f}")
        return accuracy, f1

    def preprocess(self, data):
        start_eeg = layouts[self.board_id]["eeg_start"]
        end_eeg = layouts[self.board_id]["eeg_end"]
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1]
        raw = convert_to_mne(eeg=eeg, trigger=trigger, rescale=1e6, fs=self.fs, chs=self.chs, recompute=False,
                             transpose=False)
        if BoardIds.ENOPHONE_BOARD == self.board_id:
            raw, _ = set_eeg_reference(raw, ref_channels='average')
        events = find_events(raw, stim_channel='STI', initial_event=True, shortest_event=1)

        # drop events bigger than 90
        events = events[events[:, 2] < 90]
        logging.info(f"Found events: {events.shape} {np.array(events[:, 2])}")

        if self.mode == 'train':
            self.sequence = np.unique(events[:, 2])
        elif self.mode == 'predict':
            logging.info(f"Using sequence {self.sequence}")
            trial_events = []
            # Find iteratively the same sequence of consecutive values in the events
            j = 0
            for i in range(0, len(events)):
                if events[i][2] == self.sequence[j]:
                    j += 1
                    trial_events.append(events[i])
                    if j == len(self.sequence):
                        break
                else:
                    j = 0
                    trial_events = []

            if len(trial_events) < len(self.sequence):
                raise ValueError("Not enough events found for prediction")
            events = np.array(trial_events)
            logging.info(f'Found matching events sequence: {np.array(trial_events)}')

        events[:, 2][events[:, 2] != 1] = 3
        logging.info(f"Events: {events.shape} {np.array(events[:, 2])}")


        eps = Epochs(raw, events, event_id={'T': 1, 'NT': 3}, tmin=-.1, tmax=0.9, baseline=(-.1, 0.0), preload=True)

        preprocessed = eps.get_data(picks='eeg')#[:, :, self.seg_start:self.seg_end]
        labels = eps.events[:, -1]
        logging.info(f"Data preprocessed and epochs extracted with shape {preprocessed.shape}")
        return preprocessed, labels

    def predict(self, data, proba=False, group=False):
        """
        Predict the class labels or probabilities for the input data
        :param data: numpy array of shape (n_samples, n_channels)
        :param proba: bool, whether to return class labels or probabilities
        :param group: bool, whether to group the predictions by the sequence
        :return:
        """
        try:
            X, _ = self.preprocess(data)
        except Exception as e:
            logging.error(f"Error while preprocessing data: {e}")
            return
        X = X.reshape(X.shape[0], -1)
        X = self.scaler.transform(X)
        if not group:
            return self.predict_class(X) if not proba else self.predict_probabilities(X)
        else:
            return self.group_predictions(X, proba=proba)

    def predict_class(self, X):
        """
        Compute the class labels for the input data
        :param X: numpy array of shape (n_samples, n_features)
        :return: numpy array of shape (n_samples, )
        """
        return self.model.predict(X)

    def predict_probabilities(self, X):
        """
        Compute the probability estimates for the input data
        :param X: numpy array of shape (n_samples, n_features)
        :return: numpy array of shape (n_samples, n_classes)
        """
        return self.model.predict_proba(X)

    def group_predictions(self, X, norm='z-score', proba=True):
        y = self.predict_class(X) if not proba else self.predict_probabilities(X)
        seq = self.sequence
        if proba:
            y_1 = y[:, 1]
            y_1 = normalize(y_1, method='softmax')
            y_1 = np.round(y_1, 2)
            # Determine the class with the highest probability in y_1
            output = {'class': np.argmax(y_1) + 1}  # Adding 1 to match 1-indexed classes
            for idx, prediction in enumerate(seq):
                output[str(seq[idx])] = y_1[idx]
        else:
            output = {'class': np.argmax(y) + 1}
            for idx, prediction in enumerate(seq):
                output[str(seq[idx])] = y[idx]
        return output

    def plot_roc_curve(self, n_splits=5):
        """ Plot the cross-validated ROC curve for the model"""
        plt.close()
        plot_cross_validated_roc_curve(clf=self.model, X=self.prep_X, y=self.prep_Y, pos_label=0, n_splits=n_splits,
                                       random_state=random_state)

    def plot_confusion_matrix(self, n_splits=5):
        """ Plot the cross-validated confusion matrix for the model"""
        plt.close()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        plot_cross_validated_confusion_matrix(X=self.prep_X, y=self.prep_Y, clf=self.model, cv=cv, normalize=True)

    def set_prediction_mode(self, mode=False):
        if mode:
            self.mode = 'predict'
        else:
            self.mode = 'train'
        logging.debug(f"Classifier mode set to {self.mode}")
