import time
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from typing import List, Tuple, Iterable
from algorithms.random_forest_algorithm import RandomForest


def compare_classifiers(exp_num: int, X, y, clf1, clf2) -> Tuple[float, float, float, float, float, float, np.ndarray,
                                                                 float, float, float, float, float, float, np.ndarray]:
    acc_list_clf1 = []
    f1_list_clf1 = []
    conf_mtx_list_clf1 = []
    time_list_clf1 = []
    acc_list_clf2 = []
    f1_list_clf2 = []
    conf_mtx_list_clf2 = []
    time_list_clf2 = []

    for _ in range(exp_num):
        rnd_state = np.random.randint(1, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd_state)
        acc_clf1, f1_clf1, time_clf1, conf_mtx_clf1 = eval_classifier(clf1, X_train, X_test, y_train, y_test)
        acc_clf2, f1_clf2, time_clf2, conf_mtx_clf2 = eval_classifier(clf2, X_train, X_test, y_train, y_test)

        acc_list_clf1.append(acc_clf1)
        f1_list_clf1.append(f1_clf1)
        time_list_clf1.append(time_clf1)
        conf_mtx_list_clf1.append(conf_mtx_clf1)

        acc_list_clf2.append(acc_clf2)
        f1_list_clf2.append(f1_clf2)
        time_list_clf2.append(time_clf2)
        conf_mtx_list_clf2.append(conf_mtx_clf2)

    return (round(np.mean(acc_list_clf1), 2), round(np.std(acc_list_clf1, axis=0), 2), round(np.mean(f1_list_clf1), 2),
            round(np.std(f1_list_clf1, axis=0), 2), round(np.mean(time_list_clf1), 2),
            round(np.std(time_list_clf1, axis=0), 2),
            np.round(np.sum(conf_mtx_list_clf1, axis=0) / len(conf_mtx_list_clf1)),
            round(np.mean(acc_list_clf2), 2), round(np.std(acc_list_clf2, axis=0), 2),
            round(np.mean(f1_list_clf2), 2), round(np.std(f1_list_clf2, axis=0), 2), round(np.mean(time_list_clf2), 2),
            round(np.std(time_list_clf2, axis=0), 2),
            np.round(np.sum(conf_mtx_list_clf2, axis=0) / len(conf_mtx_list_clf2)))


def eval_classifier(clf, X_train, X_test, y_train, y_test) -> Tuple[float, float, float, np.ndarray]:
    clf_start = time.time()
    clf.fit(X_train, y_train)
    clf_time = time.time() - clf_start
    y_pred = np.array(clf.predict(X_test), dtype=int)

    return (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'), clf_time,
            confusion_matrix(y_test, y_pred))


def rf_experiment_classifier_ratio(experiments_number: int, X, y, n: int, samples_percentage: float,
                                   attributes_percentage: float, classifiers: List,
                                   classifiers_ratios: List[List[float]]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for classifiers_ratio in classifiers_ratios:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)

        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(round(np.sqrt(np.mean(accuracies_std_list) ** 2 + np.std(accuracies) ** 2), 2))
        final_f1_scores_std_list.append(round(np.sqrt(np.mean(f1_scores_std_list) ** 2 + np.std(f1_scores) ** 2), 2))

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def rf_experiment_samples_percentages(experiments_number: int, X, y, n: int, samples_percentages: List[float],
                                      attributes_percentage: float, classifiers: List,
                                      classifiers_ratio: List[float]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for samples_percentage in samples_percentages:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)

        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(round(np.sqrt(np.mean(accuracies_std_list) ** 2 + np.std(accuracies) ** 2), 2))
        final_f1_scores_std_list.append(round(np.sqrt(np.mean(f1_scores_std_list) ** 2 + np.std(f1_scores) ** 2), 2))

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def rf_experiment_tree_number(experiments_number: int, X, y, n_list: List[int], samples_percentage: float,
                              attributes_percentage: float, classifiers: List, classifiers_ratio: List[float]) \
        -> Tuple[List[float], List[float], List[float], List[float], np.ndarray]:
    final_accuracies = []
    final_accuracies_std_list = []
    final_f1_scores = []
    final_f1_scores_std_list = []
    final_conf_matrices = []

    for n in n_list:
        accuracies = []
        accuracies_std_list = []
        f1_scores = []
        f1_scores_std_list = []
        conf_matrices = []
        random_forest = RandomForest(n, samples_percentage, attributes_percentage, classifiers, classifiers_ratio)

        for i in range(experiments_number):
            acc, acc_std, f1, f1_std, conf_matrix = eval_cross_validation(X, y, random_forest)
            accuracies.append(acc)
            accuracies_std_list.append(acc_std)
            f1_scores.append(f1)
            f1_scores_std_list.append(f1_std)
            conf_matrices.append(conf_matrix)

        final_accuracies.append(round(np.mean(accuracies), 2))
        final_f1_scores.append(round(np.mean(f1_scores), 2))
        final_conf_matrices.append(np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))
        # Wyliczanie złożonego odchylenia standardowego jako pierwiastek z sumy kwadratów odchyleń standardowych
        final_accuracies_std_list.append(round(np.sqrt(np.mean(accuracies_std_list) ** 2 + np.std(accuracies) ** 2), 2))
        final_f1_scores_std_list.append(round(np.sqrt(np.mean(f1_scores_std_list) ** 2 + np.std(f1_scores) ** 2), 2))

    return (final_accuracies, final_accuracies_std_list, final_f1_scores, final_f1_scores_std_list,
            np.round(np.sum(final_conf_matrices, axis=0) / len(final_conf_matrices)))


def eval_cross_validation(X, y, model, splits_number: int = 5) -> Tuple[float, float, float, float, np.ndarray]:
    accuracies = []
    f1_scores = []
    conf_matrices = []

    kf = KFold(n_splits=splits_number, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X, y):
        train_index_list = train_index.tolist()
        test_index_list = test_index.tolist()

        valid_train_indices = [idx for idx in train_index_list if idx in X.index]
        valid_test_indices = [idx for idx in test_index_list if idx in X.index]
        X_train, y_train = X.loc[valid_train_indices, :], y.loc[valid_train_indices]
        X_test, y_test = X.loc[valid_test_indices, :], y.loc[valid_test_indices]

        model.fit(X_train, y_train)
        accuracy, f1_score, conf_matrix = model.eval(X_test, y_test)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        conf_matrices.append(conf_matrix)

    return (np.mean(accuracies), np.std(accuracies, axis=0), np.mean(f1_scores),
            np.std(f1_scores, axis=0), np.round(np.sum(conf_matrices, axis=0) / len(conf_matrices)))


def plot_confusion_matrix(conf_mtx: np.ndarray, class_labels: list, class_name: str, exp_name: str):
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Confusion Matrix')
    plt.savefig(f'matrices/{class_name}_{exp_name}.png')


def format_label(val):
    if isinstance(val[0], (list, tuple)):
        return '[' + ', '.join(map(str, val)) + ']'
    else:
        return str(val)


def plot_results(x_val: List, y_val: List[float], y_std_val: List[float], x_label: str, y_label: str, class_name: str, exp_type: str):
    plt.style.use('default')
    plt.figure(figsize=(8, 6))

    if isinstance(x_val[0], Iterable):
        x_ticks = [format_label(val) for val in x_val]
    else:
        x_ticks = x_val
    plt.errorbar(x_ticks, y_val, yerr=y_std_val, fmt='o', label=x_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} = f({x_label})")
    plt.grid(True)

    plt.xscale('linear')
    plt.yscale('linear')

    file_name = f'images/{class_name}_plot_{y_label}_{exp_type}.png'
    plt.savefig(file_name)


def generate_excel_table(x_val: List, y_val: List[float], y_std_val: List[float], y2_val: List[float],
                         y_f1_val: List[float], x_label: str, y_label: str, y1_label: str, class_name: str,
                         exp_type: str):
    data = {f'{x_label}': x_val, y_label: y_val, f'{y_label}_std': y_std_val, y1_label: y2_val,
            f'{y1_label}_std': y_f1_val}
    df = pd.DataFrame(data)

    file_name = f'tables/{class_name}_table_{y_label}_{y1_label}_{exp_type}.xlsx'
    df.to_excel(file_name, index=False)


def save_classifiers_comparison_results(results_clf1: List[float], results_clf2: List[float], clf1_name: str,
                                        clf2_name: str, dataset_name: str, exp_name: str):
    data = {
        'Metric': ['Accuracy', 'Accuracy std', 'F1 Score', 'F1 Score std', 'Time', 'Time std'],
        clf1_name: [results_clf1[0], results_clf1[1], results_clf1[2], results_clf1[3], results_clf1[4], results_clf1[5]],
        clf2_name: [results_clf2[0], results_clf2[1], results_clf2[2], results_clf2[3], results_clf2[4], results_clf2[5]],
    }
    df = pd.DataFrame(data).T

    file_path = f'tables/{exp_name}_{clf1_name}_{clf2_name}_{dataset_name}.xlsx'
    df.to_excel(file_path, index=True, header=False)
