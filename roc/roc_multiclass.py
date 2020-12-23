import torchvision
from torch.autograd import Variable
import torch.optim as optim
import torch
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
from my_dataset import Xinguan, get_data
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import json
from senet.se_resnet import se_resnet50
import matplotlib.pyplot as plt
import os

class_list = ['normal', 'xingguan', 'xijunxing', 'bingdu']
def evaluate(model,testset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    model.eval() 
    targets = []
    scores = []
    id_list = []
    correct = 0
    confusion_matrix = [[0 for j in range(4)] for i in range(4)]
    for batch_i, batch_data in enumerate(dataloader):
            
        inputs = Variable(batch_data['image'].to(device),requires_grad=True)
        labels = Variable(batch_data['target'].to(device),requires_grad=False)
        id_list += batch_data['patient_id'].tolist()
        outputs = model(inputs)
        scores += outputs.data.tolist()
        targets += labels.tolist()

    return evaluate_per_patient(id_list, scores, targets)

def evaluate_per_patient(id_list, scores, targets):
    id_list = np.array(id_list)
    scores = np.array(scores)
    targets = np.array(targets)
    new_scores = []
    new_targets = []

    idx = np.argsort(id_list)
    scores = scores[idx]
    targets = targets[idx]
    id_list = id_list[idx]
    id_prev = -1
    target_prev = -1
    score_list = []
    
    for score, my_id, target in zip(scores, id_list, targets):
        if id_prev == -1:
            id_prev = my_id
            score_list.append(score)
            target_prev = target
            continue
        if my_id == id_prev:
            score_list.append(score)
        else:
            new_scores.append(np.mean(score_list, axis=0))
            new_targets.append(target_prev)
            id_prev = my_id
            score_list = [score]
            target_prev = target
    new_scores.append(np.mean(score_list, axis=0))
    new_targets.append(target_prev)
    new_scores = np.array(new_scores)
    new_targets = np.array(new_targets)
    new_predicted = np.argmax(new_scores, axis=1)

    roc = dict()
    auc = dict()
    for ii in range(4):
        auc_targets = (np.array(new_targets)==ii)
        auc_scores = np.array(new_scores)[:,ii]
        roc[class_list[ii]] = metrics.roc_curve(auc_targets, auc_scores)
        auc[class_list[ii]] = metrics.roc_auc_score(auc_targets, auc_scores)
    return auc, roc   
    

if __name__ == "__main__":
    graphTitle = str("ROC curve evaluated on patient level")
    fig1 = plt.figure(1)
    ax = plt.gca()
    plt.plot([0, 1], [0, 1], 'k--')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = se_resnet50(num_classes=4)
    model = model.to(device)    
    model = torch.nn.DataParallel(model)

    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/multi_classifier/split_info'
    checkpoint = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/multi_classifier6/checkpoints/epoch_58.pth'
    model.load_state_dict(torch.load(checkpoint))
    img_paths, targets, patient_ids, img_ids = get_data(split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    auc, roc = evaluate(model,valset)
    
    fpr, tpr, thresholds = roc['xingguan']
    label = f"COVID-19 (AUC={0.93})"
    plt.plot(fpr, tpr, color='g', lw=2, label=label)

    fpr, tpr, thresholds = roc['normal']
    label = f"Healthy (AUC={0.999})"
    plt.plot(fpr, tpr, color='r', lw=2, label=label)

    fpr, tpr, thresholds = roc['xijunxing']
    label = f"Bac. Pneu. (AUC={0.97})"
    plt.plot(fpr, tpr, color='b', lw=2, label=label)

    fpr, tpr, thresholds = roc['bingdu']
    label = f"Vir. Pneu. (AUC={0.95})"
    plt.plot(fpr, tpr, color='y', lw=2, label=label)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(graphTitle)
    plt.legend(loc='best')
    plt.savefig("multi_label_roc.png", bbox_inches=0, dpi=300)
      
