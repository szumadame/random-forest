#https://studygyaan.com/data-science/creating-a-decision-tree-using-the-id3-algorithm
import random
from scripts_and_experiments.datasets_manager import *


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return -np.sum(
        [(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])


def information_gain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    return total_entropy - weighted_entropy


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


class ID3:
    def __init__(self):
        self.tree = None

    def id3_algorithm(self, data, original_data, features, target_attribute_name, parent_node_class=None):
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
        elif len(data) == 0:
            return np.unique(original_data[target_attribute_name])[
                np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
        elif len(features) == 0:
            return parent_node_class
        else:
            parent_node_class = np.unique(data[target_attribute_name])[
                np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
            item_values = [information_gain(data, feature, target_attribute_name) for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            tree = {best_feature: {}}
            features = [i for i in features if i != best_feature]
            for value in np.unique(data[best_feature]):
                sub_data = data.where(data[best_feature] == value).dropna()
                subtree = self.id3_algorithm(sub_data, data, features, target_attribute_name, parent_node_class)
                tree[best_feature][value] = subtree
            return tree

    def fit(self, X, y):
        target_attribute_name = y.name
        features = list(X.columns)
        df = pd.concat([X, pd.DataFrame({target_attribute_name: y})], axis=1)
        self.tree = self.id3_algorithm(df, df, features, target_attribute_name)

    def predict_row(self, query, tree=None, default=1):
        if tree is None:
            tree = self.tree
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]]
                except:
                    return default
                if isinstance(result, dict):
                    return self.predict_row(query, result)
                else:
                    return result

    def predict(self, X_test):
        return X_test.apply(lambda row: self.predict_row(row), axis=1)

    def get_accuracy(self, X, y):
        target_attribute_name = y.name
        df = pd.concat([X, pd.DataFrame({target_attribute_name: y})], axis=1)
        df["classification"] = df.apply(lambda row: self.predict_row(row), axis=1)
        df["classification_correct"] = df["classification"] == df[target_attribute_name]
        return df["classification_correct"].mean()

