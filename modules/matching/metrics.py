import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score


def compute_metrics(p):
    preds, labels = map(torch.tensor, (p.predictions, p.label_ids))
    if preds.ndim > 1:
        preds = preds[0]

    preds_class = (torch.sigmoid(preds) > 0.5).int()
    labels = labels.int()

    accuracy = accuracy_score(labels, preds_class)
    precision = precision_score(labels, preds_class, zero_division=0)
    recall = recall_score(labels, preds_class, zero_division=0)
    f1 = f1_score(labels, preds_class, zero_division=0)
    auc = roc_auc_score(labels, torch.sigmoid(preds)) if len(set(labels.tolist())) > 1 else None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'combined_score': accuracy + precision + recall + f1 + auc
    }