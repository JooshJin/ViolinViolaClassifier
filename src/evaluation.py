import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def train_test_split_and_evaluate(X, y, clf, test_size=0.2, random_state=42):
    """
    Split X/y, fit classifier, return metrics dict

    If any class has fewer than 2 samples, perform a random split without stratification.
    """
    from sklearn.model_selection import train_test_split
    # Determine if stratification is possible
    counts = np.bincount(y)
    stratify_param = y if np.min(counts) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_param,
        random_state=random_state
    )
    # Train
    clf.fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f,
        'confusion_matrix': cm.tolist()
    }

def plot_confusion_matrix(cm, classes=None, normalize=False, title='Confusion matrix'):
    """
    Plot and normalize the confusion matrix
    """
    cm = np.array(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    if classes:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    else:
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
