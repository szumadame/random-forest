# Plik używany do testowania działania ID3 podczas implementacji dla naszych zbiorów danych
from datasets_manager import *
from sklearn.model_selection import train_test_split
from algorithms.id3_classifier import ID3
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def test_id3_for_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = ID3()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    classifier.get_accuracy(X_test, y_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, predictions, zero_division=1)
    print("Classification Report:")
    print(class_report)

    return accuracy, conf_matrix, class_report


print("======== DIVORCE ==========")
X, y = get_dataset_divorce()
test_id3_for_dataset(X, y)

print("\n======== CORONA ==========")
X, y = get_dataset_corona()
test_id3_for_dataset(X, y)

print("\n======== GLASS ==========")
X, y = get_dataset_glass()
test_id3_for_dataset(X, y)

print("\n======== LOAN APPROVAL ==========")
X, y = get_dataset_loan_approval()
test_id3_for_dataset(X, y)