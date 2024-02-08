import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC


class RandomForest:

    def __init__(self, n: int = 100, samples_percentage: float = 0.2, attributes_percentage: float = 0.5,
                 classifiers: List = [ID3, NBC], classifiers_ratios: List = [0.5, 0.5]):
        self.attributes = []
        self.forest = []
        self.n = n
        self.samples_percentage = samples_percentage
        self.attributes_percentage = attributes_percentage
        self.classifiers = classifiers
        self.classifiers_ratios = classifiers_ratios

    def fit(self, X_train, y_train):
        for classifier, ratio in zip(self.classifiers, self.classifiers_ratios):
            for _ in range(round(self.n * ratio)):
                self._fit_and_append_classifier(classifier, X_train, y_train)

    def predict(self, X_test) -> pd.Series:
        predictions = self._generate_predictions(X_test)
        pred_df = self._combine_predictions(predictions)
        return pd.Series(pred_df[0])

    def eval(self, X_test, y_test) -> Tuple[float, float, float]:
        y_test_np = np.array(y_test)
        y_pred_np = np.array(self.predict(X_test))
        return (accuracy_score(y_test_np, y_pred_np), f1_score(y_test_np, y_pred_np, average='macro'),
                confusion_matrix(y_test_np, y_pred_np))

    def _bagging(self, X_train, y_train):
        bagging_data = X_train.sample(frac=self.attributes_percentage, axis=1)
        self.attributes.append(bagging_data.columns)
        bagging_data['y'] = y_train
        bagging_data = bagging_data.sample(frac=self.samples_percentage)
        return bagging_data.drop(columns=['y']), bagging_data['y']

    def _fit_and_append_classifier(self, classifier, X_train, y_train):
        clf = classifier()
        X_train_bag, y_train_bag = self._bagging(X_train, y_train)
        clf.fit(X_train_bag, y_train_bag)
        self.forest.append(clf)

    def _generate_predictions(self, X_test):
        predictions = np.empty([len(self.forest), len(X_test)])
        for i in range(len(self.forest)):
            predictions[i] = np.array(self.forest[i].predict(X_test[self.attributes[i]]))
        return predictions

    def _combine_predictions(self, predictions):
        pred_df = pd.DataFrame(predictions)
        pred_df = pd.DataFrame.mode(pred_df, axis=0).T
        return pred_df