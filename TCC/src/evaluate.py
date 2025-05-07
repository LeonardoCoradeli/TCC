import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.metrics import geometric_mean_score

def compute_metrics(y_true, y_pred, y_proba=None):
    results = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }

    unique_classes = np.unique(y_true)
    if y_proba is not None and len(unique_classes) == 2:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        results['roc_auc'] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    results.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

    results['gmean'] = geometric_mean_score(y_true, y_pred)
    return results
