# Plik używany do testowania działania NBC podczas implementacji dla naszych zbiorów danych
from datasets_manager import *
from sklearn.model_selection import train_test_split
from algorithms.nbc_classifier import NBC
from sklearn import metrics


def test_nbc_for_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    nbc = NBC(1)
    nbc.fit(X_train, y_train)
    scores = nbc.score(X_test, y_test)
    y_pred = nbc.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Confusion matrix:")
    print(conf_matrix)

    return scores


print("======== DIVORCE ==========")
X, y = get_dataset_divorce()
test_nbc_for_dataset(X, y)

print("\n======== CORONA ==========")
X, y = get_dataset_corona()
test_nbc_for_dataset(X, y)

print("\n======== GLASS ==========")
X, y = get_dataset_glass()
test_nbc_for_dataset(X, y)

print("\n======== LOAN APPROVAL ==========")
X, y = get_dataset_loan_approval()
test_nbc_for_dataset(X, y)