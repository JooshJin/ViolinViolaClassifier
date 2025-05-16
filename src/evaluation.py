import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def train_test_split_and_evaluate(X, y, clf, test_size=0.2, random_state=42):
    """
    Split X/y, fit classifier, return metrics dict.
    """
    # stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # train
    clf.fit(X_train, y_train)
    # predict
    y_pred = clf.predict(X_test)
    # metrics
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f, 'confusion_matrix': cm.tolist()}
