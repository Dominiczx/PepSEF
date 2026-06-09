import torch
import numpy as np
import csv
from utils.metrics import instances_overall_metrics, label_overall_metrics, overall_metrics, binary_metrics 

def validate(args, val_dataloader, model, phase='val', save_csv=False):
    val_acc, val_pre, val_rec, val_f1 = 0, 0, 0, 0
    e2, e3 = [], []
    all_probs, all_true = [], []
    for seq, token, pssm, label, valid_lens, attention_mask in val_dataloader:
        seqs = list(seq)
        tokens = token.to(args.device)
        pssms = pssm.to(args.device)
        labels = label.to(args.device)
        attention_mask = attention_mask.to(args.device)
        # Some models (ESM2, ProtT5 variants) expect raw sequence strings as input,
        # while others (BERT-based) expect token ids. Choose based on model class name.
        mname = model.__class__.__name__.lower()
        if 'prott5' in mname:
            # ProtT5 wrapper expects (seqs, pssm, attention_mask)
            outputs, key_scores = model(seqs, pssms, attention_mask)
        elif 'esm2' in mname:
            # ESM2 wrapper expects (seqs, pssm)
            outputs, key_scores = model(seqs, pssms)
        else:
            # default: pass token ids
            outputs, key_scores = model(tokens, pssms, attention_mask)
        # keep both probability scores and binary predictions
        probs = torch.sigmoid(outputs).cpu().detach().numpy()
        pred_bin = (probs > 0.5).astype(int)
        # key_indices, key_subsequences = model.extract_key_subsequences(tokens, pssms, valid_lens, top_k=10)
        # print(key_indices, key_subsequences)
        # exit(0)
        y_true = labels.cpu().detach().numpy()
        # instance metrics use binary predictions
        evaluation = instances_overall_metrics(pred_bin, y_true)
        # label-level and AUC/AUPR use probability scores
        e2 = label_overall_metrics(probs, y_true)
        # binary_metrics expects scores; it will threshold internally to get per-class confusion
        e3.append(binary_metrics(probs, y_true))
        all_probs.append(probs)
        all_true.append(y_true)
        val_acc += evaluation['Accuracy']
        val_pre += evaluation['Precision']
        val_rec += evaluation['Recall']
        val_f1 += e2['F1'][0]

    # Compute per-class metrics on the full validation set to avoid batch-wise AUC NaN contamination.
    if len(all_probs) > 0:
        full_probs = np.concatenate(all_probs, axis=0)
        full_true = np.concatenate(all_true, axis=0)
        avg_metrics = np.array(binary_metrics(full_probs, full_true))
    else:
        e3 = np.array(e3)
        # Fallback path (should rarely happen) keeps previous behavior but uses nan-safe mean.
        avg_metrics = np.nanmean(e3, axis=0)

    avg_acc = avg_metrics[0]  # Average accuracy for each label
    avg_rec = avg_metrics[1]  # Average recall for each label
    avg_pre = avg_metrics[2]  # Average precision for each label
    avg_f1 = avg_metrics[3]   # Average F1 score for each label
    avg_mcc = avg_metrics[4]  # Average MCC for each label
    avg_auc = avg_metrics[5]  # Average AUC for each label

    # print(f"Average Accuracy: {avg_acc}")
    # print(f"Average Recall: {avg_rec}")
    # print(f"Average Precision: {avg_pre}")
    # print(f"Average F1: {avg_f1}")
    
    # Write metrics to a CSV file only if requested
    if args.save and save_csv:
        metric_names = ['acc', 'rec', 'pre', 'f1', 'mcc', 'auc']
        headers = ['AMP', 'TXP', 'ABP', 'AIP', 'AVP', 'ACP', 'AFP', 'DDV', 'CPP', 'CCC', 'APP', 'AAP', 'AHTP', 'PBP', 'QSP']
        # 按列组织数据
        data_matrix = [avg_acc, avg_rec, avg_pre, avg_f1, avg_mcc, avg_auc]  # shape: [6, 15]
        data_matrix = np.array(data_matrix)  # shape: [6, 15]
        data_matrix = data_matrix.T  # shape: [15, 6]
        with open(args.csv_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([''] + metric_names)  # 第一行：空+指标名
            for header, row in zip(headers, data_matrix):
                writer.writerow([header] + list(row))  # 每行：类别名+各指标
        
    # exit(0)
    val_acc /= len(val_dataloader)
    val_pre /= len(val_dataloader)
    val_rec /= len(val_dataloader)
    val_f1 /= len(val_dataloader)
    # print(f'{phase} acc:{val_acc}, pre:{val_pre} rec:{val_rec}')
    # print(key_scores[0])
    return val_acc, val_pre, val_rec, val_f1

def iu_validate(args, val_dataloader, model, phase='val'):
    val_acc, val_pre, val_rec, val_f1 = 0, 0, 0, 0
    for seq, scr, label in val_dataloader:
        seq, scr, labels = seq.to(args.device), scr.to(args.device), label.to(args.device)
        outputs = model(seq, scr)
        output = torch.where(outputs > 0, 1, 0).float()
        pred = output.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        evaluation = instances_overall_metrics(pred, y_true)
        e2 = label_overall_metrics(pred, y_true)
        val_acc += evaluation['Accuracy']
        val_pre += evaluation['Precision']
        val_rec += evaluation['Recall']
        val_f1 += e2['F1'][0]
    val_acc /= len(val_dataloader)
    val_pre /= len(val_dataloader)
    val_rec /= len(val_dataloader)
    val_f1 /= len(val_dataloader)
    # print(f'{phase} acc:{val_acc}, pre:{val_pre} rec:{val_rec}')
    return val_acc, val_pre, val_rec, val_f1

def validate3(args, val_dataloader, model, phase='val'):
    val_acc, val_pre, val_rec, val_f1 = 0, 0, 0, 0
    for data, pssm, label in val_dataloader:
        tokens, _, labels = data.to(args.device), pssm, label.to(args.device)
        outputs = model(tokens)
        output = torch.where(outputs > 0, 1, 0).float()
        pred = output.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        evaluation = instances_overall_metrics(pred, y_true)
        e2 = label_overall_metrics(pred, y_true)
        val_acc += evaluation['Accuracy']
        val_pre += evaluation['Precision']
        val_rec += evaluation['Recall']
        val_f1 += e2['F1'][0]
    val_acc /= len(val_dataloader)
    val_pre /= len(val_dataloader)
    val_rec /= len(val_dataloader)
    val_f1 /= len(val_dataloader)
    # print(f'{phase} acc:{val_acc}, pre:{val_pre} rec:{val_rec}')
    return val_acc, val_pre, val_rec, val_f1

def validate2(args, val_dataloader, encoder, model):
    val_acc, val_pre, val_rec = 0, 0, 0
    for data, pssm, label in val_dataloader:
        tokens, pssm, labels = data.to(args.device), pssm.to(args.device), label.to(args.device)
        encoded_X = encoder(tokens, pssm)
        outputs = []
        for i in range(len(labels[-1])):
            outputs.append(model[i](encoded_X))
        output = torch.cat(outputs, dim=-1)
        output = torch.where(output > 0, 1, 0).float()
        pred = output.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
        evaluation = instances_overall_metrics(pred, y_true)
        val_acc += evaluation['Accuracy']
        val_pre += evaluation['Precision']
        val_rec += evaluation['Recall']
    val_acc /= len(val_dataloader)
    val_pre /= len(val_dataloader)
    val_rec /= len(val_dataloader)
    print(f'test acc:{val_acc}, pre:{val_pre} rec:{val_rec}')

def find_best_threshold_by_mcc(y_true, y_pred_probs, thresholds=None):
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
     
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 500)
     
    best_threshold = 0.5
    best_score = -1
    
    from sklearn.metrics import matthews_corrcoef
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        # flatten across samples and classes
        if len(np.unique(y_pred.flatten())) < 2:
            continue # ignore all-0 or all-1 predictions
        try:
            score = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold

