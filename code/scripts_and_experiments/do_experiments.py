from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from algorithms.id3_classifier import ID3
from algorithms.nbc_classifier import NBC
from scripts_and_experiments.datasets_manager import *
from scripts_and_experiments.experients_helpers import *


# Porównanie klasyfikacji przy użyciu wybranej przez Nas gotowej i lekko przerobionej pod Nasze
# potrzeby implementacji ID3 z gotową implementacją z biblioteki sklearn
def id3_comparison(dataset_name: str):
    print('===========================================================================================================')
    print("ROZPOCZĘCIE badania porównującego wybraną przez nas implementację ID3 z implementacją z biblioteki sklearn\n")
    experiments_number = 25

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    our_id3 = ID3()
    library_id3 = DecisionTreeClassifier(criterion="entropy")

    clf1_name = 'Our_ID3'
    clf2_name = 'Library_ID3'

    (acc_1, acc_std_1, f1_1, f1_std_1, time_1, time_std_1, avg_conf_mtx_1, acc_2, acc_std_2, f1_2, f1_std_2, time_2,
     time_std_1, avg_conf_mtx_2) = compare_classifiers(experiments_number, X, y, our_id3, library_id3)

    results_clf1 = [acc_1, acc_std_1, f1_1, f1_std_1, time_1, time_std_1]
    results_clf2 = [acc_2, acc_std_2, f1_2, f1_std_2, time_2, time_std_1]

    plot_confusion_matrix(avg_conf_mtx_1, class_labels, dataset_name, clf1_name)
    plot_confusion_matrix(avg_conf_mtx_2, class_labels, dataset_name, clf2_name)

    save_classifiers_comparison_results(results_clf1, results_clf2, clf1_name, clf2_name, dataset_name, 'comparison')

    print("KONIEC badania porównującego wybraną przez nas implementację ID3 z implementacją z biblioteki sklearn")
    print("Wyniki zapisują się w folderach matrices i tables")
    print("===========================================================================================================\n")

# Porównanie klasyfikacji przy użyciu Naszej implementacji algorytmu NBC z gotową implementacją z biblioteki sklearn
def nbc_comparison(dataset_name: str):
    print('===========================================================================================================')
    print("ROZPOCZĘCIE badania porównującego naszą implementację NBC z implementacją z biblioteki sklearn\n")

    experiments_number = 25

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    our_nbc = NBC(1)
    library_nbc = CategoricalNB()

    clf1_name = 'Our_NBC'
    clf2_name = 'Library_NBC'

    (acc_1, acc_std_1, f1_1, f1_std_1, time_1, time_std_1, avg_conf_mtx_1, acc_2, acc_std_2, f1_2, f1_std_2, time_2,
     time_std_1, avg_conf_mtx_2) = compare_classifiers(experiments_number, X, y, our_nbc, library_nbc)

    results_clf1 = [acc_1, acc_std_1, f1_1, f1_std_1, time_1, time_std_1]
    results_clf2 = [acc_2, acc_std_2, f1_2, f1_std_2, time_2, time_std_1]

    plot_confusion_matrix(avg_conf_mtx_1, class_labels, dataset_name, clf1_name)
    plot_confusion_matrix(avg_conf_mtx_2, class_labels, dataset_name, clf2_name)

    save_classifiers_comparison_results(results_clf1, results_clf2, clf1_name, clf2_name, dataset_name, 'comparison')

    print("KONIEC badania porównującego naszą implementację NBC z implementacją z biblioteki sklearn")
    print("Wyniki zapisują się w folderach matrices i tables")
    print("===========================================================================================================\n")

# Porównanie wpływu liczby drzew na klasyfikację
def tree_number_influence(dataset_name: str):
    print('===========================================================================================================')
    print("ROZPOCZĘCIE badania wpływu liczby drzew na klasyfikację\n")

    experiments_number = 25
    n = [10, 20, 50, 100]
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_tree_number(experiments_number, X, y, n,
                                                                       samples_percentage, attributes_percentage,
                                                                       classifiers, classifiers_ratios)

    plot_results(n, acc, acc_std, 'n', 'Accuracy', dataset_name, 'number_of_trees')
    plot_results(n, f1, f1_std, 'n', 'F1 Score', dataset_name, 'number_of_trees')
    generate_excel_table(n, acc, acc_std, f1, f1_std, 'Number of trees', 'Accuracy', 'F1_Score',
                         dataset_name, 'number_of_trees')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'number_of_trees')

    print("KONIEC badania wpływu liczby drzew na klasyfikację")
    print("Wyniki zapisują się w folderach images, matrices i tables")
    print("===========================================================================================================\n")

# Porównanie wpływu proporcji między rodzajami klasyfikatorów na klasyfikację
def classifier_ratio_influence(dataset_name: str):
    print('===========================================================================================================')
    print("ROZPOCZĘCIE badania wpływu proporcji między rodzajami klasyfikatorów na klasyfikację\n")

    experiments_number = 25
    n = 30
    samples_percentage = 0.75
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [[0, 1], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1, 0]]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_classifier_ratio(experiments_number, X, y, n,
                                                                            samples_percentage, attributes_percentage,
                                                                            classifiers, classifiers_ratios)

    plot_results(classifiers_ratios, acc, acc_std, 'Classifiers ratio', 'Accuracy', dataset_name, 'classifiers_ratios')
    plot_results(classifiers_ratios, f1, f1_std, 'Classifiers ratio', 'F1_Score', dataset_name, 'classifiers_ratios')
    generate_excel_table(classifiers_ratios, acc, acc_std, f1, f1_std, 'Classifiers ratio', 'Accuracy', 'F1_Score',
                         dataset_name, 'classifiers_ratios')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'classifiers_ratios')

    print("KONIEC badania wpływu proporcji między rodzajami klasyfikatorów na klasyfikację")
    print("Wyniki zapisują się w folderach images, matrices i tables")
    print("===========================================================================================================\n")

# Porównanie wpływu ilości przykładów w węźle na klasyfikację
def samples_percentage_influence(dataset_name: str):
    print('===========================================================================================================')
    print("ROZPOCZĘCIE badania wpływu ilości przykładów w węźle na klasyfikację\n")

    experiments_number = 25
    n = 30
    samples_percentage = [0.2, 0.4, 0.6, 0.8]
    attributes_percentage = 0.75
    classifiers = [NBC, ID3]
    classifiers_ratios = [0.5, 0.5]

    X, y = load_proper_dataset(dataset_name)
    class_labels = get_class_labels_for_dataset(dataset_name)

    acc, acc_std, f1, f1_std, avg_conf_mtx = rf_experiment_samples_percentages(experiments_number, X, y, n,
                                                                               samples_percentage,
                                                                               attributes_percentage,
                                                                               classifiers, classifiers_ratios)

    plot_results(samples_percentage, acc, acc_std, 'Samples percentage', 'Accuracy', dataset_name, 'samples_percentage')
    plot_results(samples_percentage, f1, f1_std, 'Samples percentage', 'F1 Score', dataset_name, 'samples_percentage')
    generate_excel_table(samples_percentage, acc, acc_std, f1, f1_std, 'Samples percentage', 'Accuracy', 'F1_Score',
                         dataset_name, 'samples_percentage')
    plot_confusion_matrix(avg_conf_mtx, class_labels, dataset_name, 'samples_percentage')

    print("KONIEC badania wpływu ilości przykładów w węźle na klasyfikację")
    print("Wyniki zapisują się w folderach images, matrices i tables")
    print("===========================================================================================================\n")
