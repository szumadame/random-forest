from sklearn import metrics
import pandas as pd


def prepare_data(X_set, y_set, classColumn):
    X = pd.DataFrame(X_set)
    y = pd.DataFrame(y_set)
    df = pd.concat([X, y], axis=1)
    X = df.loc[:, df.columns != classColumn]
    y = df.loc[:, df.columns == classColumn]
    return X, y


class NBC:

    def __init__(self, alpha=None):
        self.alpha = 1.0 if alpha is None else alpha
        self.labels = None
        self.attributes = None
        self.valuesPerAttribute = {}
        self.conditionalProbabilities = {}
        self.pLabel = {}
        self.nLabelOccurrences = {}
        self.labelColumn = None

    def count_conditional_probabilities(self, df):
        for label in self.labels:
            df_for_label = df[df[self.labelColumn] == label]
            self.nLabelOccurrences[label] = len(df_for_label)
            self.pLabel[label] = (self.nLabelOccurrences[label] + self.alpha) / (
                    len(df.index) + self.alpha * len(self.labels))

            for attribute in self.attributes:
                uniqueAttributes = df[attribute].unique().tolist()
                for attributeValue in uniqueAttributes:
                    attrForClassOccurrences = len(df_for_label[df_for_label[attribute] == attributeValue])
                    key = (label, attribute, attributeValue)
                    self.conditionalProbabilities[key] = (
                        attrForClassOccurrences + self.alpha) / (
                            len(df_for_label) + self.alpha *
                            self.valuesPerAttribute[attribute])

    def fit(self, X_train, y_train):
        self.labelColumn = y_train.name
        X_train, y_train = prepare_data(X_train, y_train, self.labelColumn)
        df = X_train.join(y_train)

        # labels
        self.labels = y_train[self.labelColumn].unique()

        # attributes
        self.attributes = X_train.keys().tolist()

        for attr in self.attributes:
            self.valuesPerAttribute[attr] = X_train[attr].nunique()

        self.count_conditional_probabilities(df)

    def predict(self, X):
        return X.apply(lambda row: self.predict_row(row.values.flatten().tolist()), axis=1)

    def predict_row(self, row):
        p = float('-inf')
        predictedLabel = 'undefined'

        for label in self.labels:
            pLabel = self.pLabel[label]
            for i in range(len(self.attributes)):
                attr_value = row[i]
                key = (label, self.attributes[i], attr_value)

                if key in self.conditionalProbabilities:
                    pLabel = pLabel * self.conditionalProbabilities[key]
                else:
                    possibleValues = float(self.valuesPerAttribute[self.attributes[i]])
                    pLabel = pLabel * self.alpha / (self.nLabelOccurrences[label] + possibleValues * self.alpha)
            if pLabel > p:
                predictedLabel = label
                p = pLabel
        return predictedLabel

    def score(self, X, y):
        y_pred = self.predict(X)
        y = y.values.flatten().tolist()
        acc = metrics.accuracy_score(y, y_pred)
        f1 = metrics.f1_score(y, y_pred, average='macro')
        conf_matrix = metrics.confusion_matrix(y, y_pred)
        return acc, f1, conf_matrix
