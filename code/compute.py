import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,confusion_matrix
)
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import time
import time as t
from load import DrugDataset, DrugDataLoader
from model import model
import pandas as pd
import dgl
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.nn.functional as F



def do_compute(batch, device, model):
    tri, label = batch
    tri = [tensor.to(device=device) for tensor in tri]
    pred,_,_ = model(tri)

    return pred, label

def do_compute_metrics(pred, target, bord = False):
    pred = np.concatenate(pred)
    target = np.concatenate(target)

    # 获取预测类别
    pred_labels = np.argmax(pred, axis=1)

    # Accuracy
    acc = accuracy_score(target, pred_labels)

    # Precision, Recall, F1 (macro 平均)
    pre = precision_score(target, pred_labels, average="macro")
    rec = recall_score(target, pred_labels, average="macro")
    f1 = f1_score(target, pred_labels, average="macro")

    # AUC (需要对每一类计算，然后取平均值)
    n_classes = 86

    target_one_hot = np.eye(n_classes)[target]  # 将 target 转为 one-hot

    # auc = roc_auc_score(target_one_hot, pred)
    # print('auc',auc)
    idx = np.argwhere(np.all(np.array(target_one_hot)[..., :] == 0, axis=0))
    target_one_hot = np.delete(target_one_hot, idx, axis=1)
    pred = np.delete(pred, idx, axis=1)
    aupr = average_precision_score(target_one_hot, pred)
    metrics_per_class = []
    for class_idx in range(n_classes):
        if np.sum(target_one_hot[:, class_idx]) > 0:  # Avoid empty classes
            # Extract true labels and predictions for the current class
            true_binary = target_one_hot[:, class_idx]
            pred_binary = pred[:, class_idx]

            # Compute metrics for the current class
            precision = precision_score(true_binary, pred_labels == class_idx, zero_division=0)
            recall = recall_score(true_binary, pred_labels == class_idx, zero_division=0)
            f1 = f1_score(true_binary, pred_labels == class_idx, zero_division=0)
            aupr = average_precision_score(true_binary, pred_binary)

            metrics_per_class.append({
                "Class": class_idx,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "AUPR": aupr,
            })
        else:
            metrics_per_class.append({
                "Class": class_idx,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-Score": 0.0,
                "AUPR": 0.0,
            })

    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics_per_class)
    df_metrics.to_csv('radar/induc/SGT_each_type.csv', index=False)
    
    if bord:
        print("Confusion Matrix:")
        print(confusion_matrix(target, pred_labels))
        print("\nClassification Report:")
        print(classification_report(target, pred_labels))

    return acc, 0, f1, pre, rec, aupr

def train(train_data, val_data, args, model, optimizer, device):
    """
    训练模型。
    """
    batch_size, epoches, num_workers, fold, model_path,task = args

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_data_loader = DrugDataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    # loss_fn = FocalLoss()
    loss_fn = nn.CrossEntropyLoss()
    result = []
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    # scheduler = None
    with torch.no_grad():
        val_probas_pred, val_ground_truth = [], []
        for batch_idx, batch in enumerate(tqdm(val_data_loader, ncols=80, leave=False)):
            model.eval()
            probas_pred, ground_truth = do_compute(batch, device, model)
            val_probas_pred.append(F.softmax(probas_pred).detach().cpu().numpy())
            val_ground_truth.append(np.array(ground_truth))
        result = do_compute_metrics(val_probas_pred, val_ground_truth)
    max_acc = result[0]
    print(max_acc)
    save_result(model_path, fold, f'epoch {-1}', result, 'result/val.csv')
    ######################################### train #############################################################
    for i in range(epoches):
        iter = 0
        start = t.time()
        train_loss = 0
        train_probas_pred, train_ground_truth = [], []
        for batch_idx, batch in enumerate(tqdm(train_data_loader, ncols=80, leave=False)):
            model.train()
            optimizer.zero_grad()
            pred_logits,label= do_compute(batch, device, model)
            loss = loss_fn(pred_logits, torch.tensor(label, dtype=torch.long, device=pred_logits.device))
            train_probas_pred.append(F.softmax(pred_logits).detach().cpu().numpy())
            train_ground_truth.append(np.array(label))
            if batch_idx % 50 == 0 and batch_idx != 0:
                print(f'Batch: {batch_idx} loss: {loss.item():.4f}')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iter += 1

        train_loss /= iter

        with torch.no_grad():
            if i % 10 == 0 :
                result = do_compute_metrics(train_probas_pred, train_ground_truth,bord=True)
            else:
                result = do_compute_metrics(train_probas_pred, train_ground_truth)

            save_result(model_path, fold, f'epoch {i}', result, 'result/train.csv')
            print(f'Epoch: {i} ({t.time() - start:.4f}s), train_loss: {train_loss:.4f}, train_acc: {result[0]:.4f}')

            val_probas_pred, val_ground_truth = [], []
            for batch_idx, batch in enumerate(tqdm(val_data_loader, ncols=80, leave=False)):
                model.eval()
                probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(F.softmax(probas_pred).detach().cpu().numpy())
                val_ground_truth.append(np.array(ground_truth))

            result = do_compute_metrics(val_probas_pred,val_ground_truth)

            val_acc = result[0]
            save_result(model_path, fold, f'epoch {i}', result, 'result/val.csv')
            print(f'epoch: {i}, val_acc: {result[0]:.4f}, val_roc: {result[1]:.4f}')
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, model_path)

            print(f'max_acc: {max_acc:.4f}')

        if scheduler:
            # print('scheduling')
            scheduler.step()


def test(test_data, args, model, device, filename):
    batch_size, epoches, num_workers, fold, model_path,task = args
    test_probas_pred = []
    test_ground_truth = []
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size, num_workers=3)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_data_loader, ncols=50, leave=False)):
            model.eval()
            probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(F.softmax(probas_pred).detach().cpu().numpy())
            test_ground_truth.append(np.array(ground_truth))
        result = do_compute_metrics(test_probas_pred, test_ground_truth)
        save_result(model_path,fold,'test',result,filename)

def save_result(model_path,fold,epoch,result, file):
    import os
    column = ['time',"model", 'fold','epoch',"Accuracy", "AUC", 'F1', 'precision', 'recall', 'aupr']
    current_time = time.strftime('%Y.%m.%d %H:%M%S', time.localtime(time.time()))
    # 检查文件是否存在，如果不存在则创建一个空的 CSV 文件并写入列名
    if not os.path.isfile(file):
        pd.DataFrame(columns=column).to_csv(file, index=False)

    list_ = [current_time, model_path, fold, epoch]
    list_.extend(list(result))
    # print(list_)
    test = pd.DataFrame(columns=column, data=[list_])
    test.to_csv(file, mode='a', header=False, index=False)


from sklearn.model_selection import StratifiedKFold
import numpy as np


# def split_train_valid(data, n_splits=5, test_ratio=0.1, val_ratio=None, seed=42):
#     """
#     执行5折交叉验证，返回每一折的训练集、验证集和测试集。
#
#     参数：
#     - data: 输入数据，必须是二维数组（或矩阵），最后一列为标签。
#     - n_splits: 划分为几折交叉验证，默认为5折。
#     - test_ratio: 测试集比例。
#     - val_ratio: 验证集比例，若为 None，则使用训练集的剩余部分作为验证集。
#     - seed: 随机种子，用于控制数据划分的随机性。
#
#     返回：
#     - 返回一个包含每一折训练集、验证集和测试集的列表。
#     """
#     data = np.array(data)
#     assert val_ratio + test_ratio < 1, "验证集和测试集比例之和不能超过 1"
#
#     # 划分测试集，保证测试集是10%
#     first_split = StratifiedKFold(n_splits=1, test_size=test_ratio, shuffle=True, random_state=seed)
#     for train_val_index, test_index in first_split.split(data, data[:, -1]):
#         train_val_data = data[train_val_index]
#         test_data = data[test_index]
#
#     # 计算验证集和训练集的划分比例
#     if val_ratio is not None:
#         second_split = StratifiedKFold(n_splits=1, test_size=val_ratio / (1 - test_ratio), shuffle=True,
#                                        random_state=seed)
#         for train_index, val_index in second_split.split(train_val_data, train_val_data[:, -1]):
#             train_data = train_val_data[train_index]
#             val_data = train_val_data[val_index]
#     else:
#         train_data = train_val_data
#         val_data = train_val_data
#
#     # 保存每一折的划分结果
#     splits = []
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#
#     for train_index, val_index in skf.split(train_data, train_data[:, -1]):
#         train_fold = train_data[train_index]
#         val_fold = train_data[val_index]
#         splits.append({
#             'train': train_fold,
#             'val': val_fold,
#             'test': test_data
#         })
#
#     return splits

def split_train_valid(data, seed=42, val_ratio=None, test_ratio=0.1):

    data = np.array(data)


    # 首先划分出 90% 的数据 (train+val) 和 10% 的数据 (test)
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_index, test_index = next(iter(first_split.split(X=data, y=data[:, -1])))

    train_val_data = data[train_val_index]
    test_data = data[test_index]
    if val_ratio is not None:

        # 在 90% 的数据中，再划分出 80% (train) 和 10% (val)
        second_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / (1 - test_ratio), random_state=seed)
        train_index, val_index = next(iter(second_split.split(X=train_val_data, y=train_val_data[:, -1])))

        train_data = train_val_data[train_index]
        val_data = train_val_data[val_index]

        # 转换为指定格式
        train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_data]
        val_tup = [(tup[0], tup[1], int(tup[2])) for tup in val_data]
        test_tup = [(tup[0], tup[1], int(tup[2])) for tup in test_data]

        return train_tup, val_tup, test_tup

    else:
        train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_val_data]
        test_tup = [(tup[0], tup[1], int(tup[2])) for tup in test_data]

        return train_tup,  test_tup
