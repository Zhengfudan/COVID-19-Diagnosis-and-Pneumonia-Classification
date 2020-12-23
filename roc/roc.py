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

def evaluate(model,testset, thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    model.eval() 
    
    scores = []
    targets = []
    id_list = []
    img_id_list = []
    sig = nn.Sigmoid()
    
    for batch_i, batch_data in enumerate(dataloader):
            
        inputs = Variable(batch_data['image'].to(device),requires_grad=True)
        labels = Variable(batch_data['target'].to(device),requires_grad=False)
        id_list += batch_data['patient_id'].tolist()
        img_id_list += batch_data['img_id'].tolist()
        
        outputs = model(inputs)
        
        predicted = outputs.data        
        predicted = sig(predicted)
        scores += predicted.tolist()
        targets += labels.tolist()
    return evaluate_per_patient(id_list, scores, targets, thresh)
    
def evaluate_per_patient(id_list, scores, targets, thresh=0.5):

    id_list = np.array(id_list).reshape(-1)
    scores = np.array(scores).reshape(-1)
    targets = np.array(targets).reshape(-1)
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
            new_scores.append(np.mean(score_list))
            new_targets.append(target_prev)
            id_prev = my_id
            score_list = [score]
            target_prev = target
    new_scores.append(np.mean(score_list))
    new_targets.append(target_prev)
    new_scores = np.array(new_scores)
    new_targets = np.array(new_targets)
    fpr, tpr, thresholds = metrics.roc_curve(new_targets, new_scores)
    return fpr, tpr

if __name__ == "__main__":
    graphTitle = str("ROC curve evaluated on patient level")
    fig1 = plt.figure(1)
    ax = plt.gca()
    plt.plot([0, 1], [0, 1], 'k--')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/data/data'
    model = se_resnet50(num_classes=1)
    model = model.to(device)    
    model = torch.nn.DataParallel(model)

    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier3_2/split_info'
    checkpoint = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier3_3/checkpoints/epoch_1.pth'
    model.load_state_dict(torch.load(checkpoint))
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    fpr, tpr = evaluate(model,valset, 0.5)
    plt.plot(fpr, tpr, color='r', lw=2, label='COVID-19/Healthy(AUC=0.99)')

    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier4_2/split_info'
    checkpoint = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier4_3/checkpoints/epoch_169.pth'
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    model.load_state_dict(torch.load(checkpoint))
    fpr, tpr = evaluate(model,valset, 0.5)
    plt.plot(fpr, tpr, color='g', lw=2, label='COVID-19/Bac. Pneu.(AUC=0.94)')

    checkpoint = '/GPUFS/nsccgz_ywang_2/AA_weifeng/covid19/SENET/checkpoints/epoch_21.pth'
    split_info_path = '/GPUFS/nsccgz_ywang_2/AA_weifeng/covid19/SENET/split_info'
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    model.load_state_dict(torch.load(checkpoint))
    fpr, tpr = evaluate(model,valset, 0.5)
    plt.plot(fpr, tpr, color='b', lw=2, label='COVID-19/Vir. Pneu.(AUC=0.91)')

    checkpoint = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier5/checkpoints/epoch_23.pth'
    split_info_path = '/GPUFS/nsccgz_ywang_2/wujiahao-deeplearning/xingguan/xingguan_classifier5/split_info'
    img_paths, targets, patient_ids, img_ids = get_data(root_path, split_info_path)
    valset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])
    model = torchvision.models.resnet50(num_classes=1)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint))
    fpr, tpr = evaluate(model,valset, 0.5)
    plt.plot(fpr, tpr, color='c', lw=2, label='COVID-19/All others.(AUC=0.92)')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(graphTitle)
    plt.legend(loc='best')
    plt.savefig("per_patient_roc.png", bbox_inches=0, dpi=300)
      
