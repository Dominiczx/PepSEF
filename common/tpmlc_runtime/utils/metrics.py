import numpy as np
from sklearn.metrics import roc_auc_score, auc, accuracy_score, precision_recall_curve, f1_score, balanced_accuracy_score, \
    recall_score, precision_score, matthews_corrcoef, multilabel_confusion_matrix, roc_curve
from sklearn.metrics import hamming_loss
import os
import pandas as pd

def instances_overall_metrics(y_pred: np.array, y_true: np.array):
    """
    计算样本层面的整体评价指标
    """
    y_pred_cls = y_pred

    n, m = y_true.shape

    # Hamming Loss
    HLoss = hamming_loss(y_true, y_pred_cls)

    # Instance-based (sample-wise) Jaccard / Accuracy: mean over samples of |Y∩Z|/|Y∪Z|
    ACCs = []
    for i in range(n):
        inter = np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1))
        union = np.sum((y_pred_cls[i] == 1) | (y_true[i] == 1))
        if union == 0:
            # both empty => perfect match for this sample
            ACCs.append(1.0)
        else:
            ACCs.append(inter / union)
    ACC = float(np.mean(ACCs))

    # Instance-based Precision: average of per-sample precision (TP / predicted_pos)
    precisions = []
    for i in range(n):
        tp = np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1))
        pred_pos = np.sum(y_pred_cls[i] == 1)
        if pred_pos == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / pred_pos)
    Precision = float(np.mean(precisions))

    # Instance-based Recall: average of per-sample recall (TP / true_pos)
    recalls = []
    for i in range(n):
        tp = np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1))
        true_pos = np.sum(y_true[i] == 1)
        if true_pos == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / true_pos)
    Recall = float(np.mean(recalls))

    # Absolute ture
    AT = 0
    for i in range(n):
        if(np.all(y_pred_cls[i] == y_true[i])):
            AT += 1
    AT /= n

    df = {'HLoss': HLoss, 'Accuracy': ACC, 'Precision': Precision, 'Recall': Recall, 'Absolute true': AT}

    return df

def label_overall_metrics(y_pred: np.array, y_true: np.array):
    """
    Compute per-class and aggregated (macro/micro) metrics.

    y_pred may be either probability scores or binary predictions. We accept scores
    and threshold them internally for metrics that require binary predictions.
    Returns a dict where each entry like 'F1' is [macro, micro].
    """
    # Convert to numpy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # default threshold for converting scores -> binary
    threshold = 0.5
    # Heuristic: treat as scores if dtype is floating or there are more than 2 unique values
    is_float = np.issubdtype(y_pred.dtype, np.floating)
    try:
        unique_vals = np.unique(y_pred)
    except Exception:
        unique_vals = []
    if is_float or (unique_vals.size > 2) or (unique_vals.size == 2 and not set(unique_vals).issubset({0, 1})):
        y_scores = y_pred.astype(float)
        y_pred_cls = (y_scores >= threshold).astype(int)
    else:
        y_scores = y_pred.astype(float)
        y_pred_cls = y_pred.astype(int)

    n_samples, n_class = y_true.shape

    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []

    TP = FP = TN = FN = 0

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_scores[:, c]

        tp = int(np.sum(np.logical_and(y_t == 1, y_c == 1)))
        tn = int(np.sum(np.logical_and(y_t == 0, y_c == 0)))
        fp = int(np.sum(np.logical_and(y_t == 0, y_c == 1)))
        fn = int(np.sum(np.logical_and(y_t == 1, y_c == 0)))

        TP += tp; TN += tn; FP += fp; FN += fn

        # per-class metrics (use safe zero_division)
        try:
            F1 = f1_score(y_t, y_c, zero_division=0)
        except Exception:
            F1 = 0.0
        try:
            ACC = accuracy_score(y_t, y_c)
        except Exception:
            ACC = 0.0

        try:
            AUC = roc_auc_score(y_t, y_p)
        except Exception:
            AUC = np.nan

        try:
            precision, recall, _ = precision_recall_curve(y_t, y_p)
            AUPR = auc(recall, precision)
        except Exception:
            AUPR = np.nan

        try:
            BACC = balanced_accuracy_score(y_t, y_c)
        except Exception:
            BACC = np.nan

        try:
            MCC = matthews_corrcoef(y_t, y_c)
        except Exception:
            MCC = np.nan

        try:
            Recall = recall_score(y_t, y_c, zero_division=0)
        except Exception:
            Recall = 0.0
        try:
            Precision = precision_score(y_t, y_c, zero_division=0)
        except Exception:
            Precision = 0.0

        res_mcc.append(MCC)
        res_acc.append(ACC)
        res_auc.append(AUC)
        res_aupr.append(AUPR)
        res_recall.append(Recall)
        res_precision.append(Precision)
        res_f1.append(F1)
        res_bacc.append(BACC)

    # Micro metrics computed from accumulated TP/TN/FP/FN
    Precision_micro = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall_micro = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    BACC_micro = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2 if (TP + FN) > 0 and (TN + FP) > 0 else np.nan

    # AUC and AUPR micro (use sklearn functions where possible)
    try:
        AUC_micro = roc_auc_score(y_true, y_scores, average='micro')
    except Exception:
        AUC_micro = np.nan
    try:
        from sklearn.metrics import average_precision_score
        AUPR_micro = average_precision_score(y_true, y_scores, average='micro')
    except Exception:
        AUPR_micro = np.nan

    try:
        F1_micro = f1_score(y_true, y_pred_cls, average='micro', zero_division=0)
    except Exception:
        F1_micro = np.nan

    # macro averages (use nanmean for metrics that may contain nan)
    try:
        AUC_macro = float(np.nanmean(res_auc))
    except Exception:
        AUC_macro = np.nan
    try:
        AUPR_macro = float(np.nanmean(res_aupr))
    except Exception:
        AUPR_macro = np.nan

    ACC_macro = float(np.mean(res_acc)) if len(res_acc) > 0 else np.nan
    MCC_macro = float(np.nanmean(res_mcc)) if len(res_mcc) > 0 else np.nan
    Precision_macro = float(np.mean(res_precision)) if len(res_precision) > 0 else np.nan
    Recall_macro = float(np.mean(res_recall)) if len(res_recall) > 0 else np.nan
    BACC_macro = float(np.nanmean(res_bacc)) if len(res_bacc) > 0 else np.nan
    F1_macro = float(np.mean(res_f1)) if len(res_f1) > 0 else np.nan

    df = {
        'Accuracy': [ACC_macro, ACC_macro if np.isnan(ACC_macro) else np.mean(res_acc)],
        'BACC': [BACC_macro, BACC_micro],
        'AUC': [AUC_macro, AUC_micro],
        'MCC': [MCC_macro, np.nan],
        'AUPR': [AUPR_macro, AUPR_micro],
        'F1': [F1_macro, F1_micro],
        'Precision': [Precision_macro, Precision_micro],
        'Recall': [Recall_macro, Recall_micro]
    }
    return df


def binary_metrics(y_pred: np.array, y_true: np.array):
    """
    计算每一类的准确率，精度, 召回率, MCC
    Args:
        y_pred: 预测得分, [n_samlpes, n_class]
        y_true: 真实类别, [n_samlpes, n_class]
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    # Accept either probability scores or binary predictions. If scores, threshold at 0.5.
    y_pred = np.array(y_pred)
    if y_pred.dtype == float or (np.unique(y_pred).size > 2):
        y_pred_cls = (y_pred >= 0.5).astype(int)
    else:
        y_pred_cls = y_pred.astype(int)
    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []
    res_rkcc = []

    # multi-label confusion matrics (n_class)
    mcm = multilabel_confusion_matrix(y_true, y_pred_cls)

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        # use safe zero_division handling (0) so F1/precision/recall are not artificially 1
        F1 = f1_score(y_t, y_c, zero_division=0)
        ACC = accuracy_score(y_t, y_c)
        try:
            AUC = roc_auc_score(y_t, y_p)
        except Exception:
            AUC = np.nan

        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        # compute MCC and PR-derived metrics safely
        MCC = matthews_corrcoef(y_t, y_c) if (len(np.unique(y_t)) > 1 or len(np.unique(y_c)) > 1) else 0.0
        Recall = recall_score(y_t, y_c, zero_division=0)
        Precision = precision_score(y_t, y_c, zero_division=0)

        res_mcc.append(round(MCC, 3))
        res_acc.append(round(ACC, 3))
        res_auc.append(round(AUC, 3) if not np.isnan(AUC) else np.nan)
        res_recall.append(round(Recall, 3))
        res_precision.append(round(Precision, 3))
        res_f1.append(round(F1, 3))
    return [res_acc, res_recall, res_precision, res_f1, res_mcc, res_auc]  # , res_aupr, res_bacc, res_rkcc]
    # df = pd.DataFrame({'ACC': res_acc, 'BACC': res_bacc,'AUC': res_auc, 'MCC': res_mcc, 'AUPR': res_aupr, 'F1': res_f1,
                    #    'Precision': res_precision, 'Recall': res_recall, 'Rkcc': res_rkcc})
    # return df


def overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    综合评价多标签分类任务
    """
    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别


    HLoss = hamming_loss(y_true, y_pred_cls)

    # Calculate metrics globally by counting the total true positives,false negatives and false positives.
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    F1_macro = f1_score(y_true, y_pred_cls, average='macro')

    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    # This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

    F1_weighted = f1_score(y_true, y_pred_cls, average='weighted')

    df = pd.DataFrame({'HLoss': [HLoss], 'F1_micro': [F1_micro], 'F1_macro': [F1_macro], 'F1_weighted': [F1_weighted]})
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

    return df
