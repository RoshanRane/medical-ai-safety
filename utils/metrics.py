import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve, confusion_matrix


# def get_accuracy(y_hat, y, se=False):
#     if y.dim() == 2:
#         accuracy = ((y_hat.sigmoid() >.5).long() == y).float().mean().item()
#     else:
#         accuracy = (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
#     if se:
#         se = np.sqrt(accuracy * (1 - accuracy) / len(y))
#         return accuracy, se
#     return accuracy
## ROSH computes balanced accuracy instead
def get_accuracy(y_hat, y, se=False):
    if y.dim() == 2:
        # Binary classification
        preds = (y_hat.sigmoid() > 0.5).long().view(-1)
        y_flat = y.view(-1)
    else:
        # Multiclass classification
        preds = y_hat.argmax(dim=1)
        y_flat = y

    classes = torch.unique(y_flat)
    per_class_acc = []
    per_class_n = []

    for c in classes:
        mask = (y_flat == c)
        n_c = mask.sum().item()
        if n_c > 0:
            class_acc = (preds[mask] == c).float().mean().item()
            per_class_acc.append(class_acc)
            per_class_n.append(n_c)

    balanced_acc = np.mean(per_class_acc)

    if se:
        # SE of balanced accuracy: sqrt(sum(p_i(1-p_i)/n_i)) / k
        # where k is the number of classes
        k = len(per_class_acc)
        var_sum = sum(p * (1 - p) / n for p, n in zip(per_class_acc, per_class_n))
        se_val = np.sqrt(var_sum) / k
        return balanced_acc, se_val
    return balanced_acc


def get_f1(y_hat, y):
    pred = (y_hat.sigmoid() >.5).long()if y.dim() == 2 else y_hat.argmax(dim=1)
    return f1_score(pred.detach().cpu(), y.detach().cpu(), average='macro')


def get_auc(y_hat, y):
    pred = (y_hat.sigmoid() >.5).long().detach().cpu().numpy() if y.dim() == 2 else y_hat.softmax(1).detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    try:
        if y_hat.shape[1] > 2:
            auc = roc_auc_score(target, pred, multi_class='ovo', labels=range(pred.shape[1]))
        else:
            auc = roc_auc_score(target, pred[:, 1])
    except:
        auc = torch.tensor(0.0)
    return auc


def get_auc_label(y_true, model_outs, label):
    if y_true.dim() == 2:
        fpr, tpr, _ = roc_curve(y_true.numpy()[:,label], model_outs.numpy()[:,label])
    else:
        fpr, tpr, _ = roc_curve(y_true.numpy(), model_outs.numpy()[:, label], pos_label=label)
    return auc(fpr, tpr)

def get_fpr_label(y_true, model_preds, label):
    print(label)
    cm = confusion_matrix(y_true[:,label], model_preds[:,label], labels=(0,1))
    fp = cm[0, 1]
    tn = cm[0, 0]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr