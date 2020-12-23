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
from my_dataset import Xinguan
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import prettytable as pt
from my_dataset import Xinguan, get_data

parser.add_argument("--data_path", type=str, default='', help="img path of test data")
parser.add_argument("--checkpoints_path", type=str, default='checkpoints/Best_model.pth', help="best model")
opt = parser.parse_args()


split_info_path = opt.data_path
checkpoints = opt.checkpoints_path


class_list = ['normal', 'xingguan', 'xijunxing', 'bingdu']
# load data

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
    # 0~3 是各类别，4是总
    tp = [0 for i in range(5)]
    tn = [0 for i in range(5)]
    fp = [0 for i in range(5)]
    fn = [0 for i in range(5)]
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
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        for i, j in zip(predicted, labels):
            confusion_matrix[i.item()][j.item()] += 1
        for ii in range(4):
            tp[ii] += ((predicted == ii) * (labels == ii)).sum().item()
            fp[ii] += ((predicted == ii) * (labels != ii)).sum().item()
            tn[ii] += ((predicted != ii) * (labels != ii)).sum().item()
            fn[ii] += ((predicted != ii) * (labels == ii)).sum().item()
    for ii in range(4):
        tp[4] += tp[ii]
        fp[4] += fp[ii]
        tn[4] += tn[ii]
        fn[4] += fn[ii]

    # confusion matrix
    print(f'per image:')
    tb = pt.PrettyTable( ["predicted\\label", 'normal', 'xingguan', 'xijunxing', 'bingdu'])
    for i in range(4):
        tb.add_row([class_list[i]] + confusion_matrix[i])
    print(tb)

    precision = dict()
    recall = dict()
    F1score = dict()
    auc = dict()
    acc = correct/len(testset)
    for ii in range(4):
        precision[class_list[ii]] = tp[ii] / (fp[ii] + tp[ii] + 1e-6)
        recall[class_list[ii]] = tp[ii] / (fn[ii] + tp[ii] + 1e-6)
        F1score[class_list[ii]] = 2 * precision[class_list[ii]] * recall[class_list[ii]] / (precision[class_list[ii]] + recall[class_list[ii]] + 1e-6)
        auc_targets = (np.array(targets)==ii)
        auc_scores = np.array(scores)[:,ii]
        auc[class_list[ii]] = metrics.roc_auc_score(auc_targets, auc_scores)    
    
    print(f'acc: {acc}')
    for ii in range(4):
        print(f'class {class_list[ii]}: tp:{tp[ii]}, fp:{fp[ii]}, tn:{tn[ii]}, fn:{fn[ii]}')  
        print(f'P:{precision[class_list[ii]]}, R:{recall[class_list[ii]]}, F1:{F1score[class_list[ii]]}, auc:{auc[class_list[ii]]}')
        print('-'*20)

    precision['macro'] = recall['macro'] = F1score['macro'] = auc['macro'] = 0
    for ii in range(4):
        precision['macro'] += precision[class_list[ii]]
        recall['macro'] += recall[class_list[ii]]
        F1score['macro'] += F1score[class_list[ii]]
        auc['macro'] += auc[class_list[ii]]
    precision['macro'] /= 4
    recall['macro'] /= 4
    F1score['macro'] /= 4
    auc['macro'] /= 4
    
    print(f'total: tp:{tp[4]}, fp:{fp[4]}, tn:{tn[4]}, fn:{fn[4]}')
    print(f"macro:  P:{precision['macro']}, R:{recall['macro']}, F1:{F1score['macro']}, auc:{auc['macro']}")

    per_img_res = dict()
    per_img_res['precision'] = precision
    per_img_res['recall'] = recall
    per_img_res['acc'] = acc
    per_img_res['F1score'] = F1score
    per_img_res['auc'] = auc
    
    return per_img_res, evaluate_per_patient(id_list, scores, targets)

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

    # confusion matrix
    print(f'per patient:')
    confusion_matrix = [[0 for j in range(4)] for i in range(4)]
    for i, j in zip(new_predicted, new_targets):
        confusion_matrix[i][j] += 1
    tb = pt.PrettyTable( ["predicted\\label", 'normal', 'xingguan', 'xijunxing', 'bingdu'])
    for i in range(4):
        tb.add_row([class_list[i]] + confusion_matrix[i])
    print(tb)

    tp = [0 for i in range(5)]
    tn = [0 for i in range(5)]
    fp = [0 for i in range(5)]
    fn = [0 for i in range(5)]
    for ii in range(4):
        tp[ii] += ((new_predicted == ii) * (new_targets == ii)).sum().item()
        fp[ii] += ((new_predicted == ii) * (new_targets != ii)).sum().item()
        tn[ii] += ((new_predicted != ii) * (new_targets != ii)).sum().item()
        fn[ii] += ((new_predicted != ii) * (new_targets == ii)).sum().item()
    for ii in range(4):
        tp[4] += tp[ii]
        fp[4] += fp[ii]
        tn[4] += tn[ii]
        fn[4] += fn[ii]

    precision = dict()
    recall = dict()
    F1score = dict()
    auc = dict()
    correct = (new_predicted == new_targets).sum().item()
    acc = correct / new_targets.shape[0]
    for ii in range(4):
        precision[class_list[ii]] = tp[ii] / (fp[ii] + tp[ii] + 1e-6)
        recall[class_list[ii]] = tp[ii] / (fn[ii] + tp[ii] + 1e-6)
        F1score[class_list[ii]] = 2 * precision[class_list[ii]] * recall[class_list[ii]] / (precision[class_list[ii]] + recall[class_list[ii]] + 1e-6)
        auc_targets = (np.array(new_targets)==ii)
        auc_scores = np.array(new_scores)[:,ii]
        auc[class_list[ii]] = metrics.roc_auc_score(auc_targets, auc_scores)    
    
    print(f'acc: {acc}')
    for ii in range(4):
        print(f'class {class_list[ii]}: tp:{tp[ii]}, fp:{fp[ii]}, tn:{tn[ii]}, fn:{fn[ii]}')  
        print(f'P:{precision[class_list[ii]]}, R:{recall[class_list[ii]]}, F1:{F1score[class_list[ii]]}, auc:{auc[class_list[ii]]}')
        print('-'*20)

    precision['macro'] = recall['macro'] = F1score['macro'] = auc['macro'] = 0
    for ii in range(4):
        precision['macro'] += precision[class_list[ii]]
        recall['macro'] += recall[class_list[ii]]
        F1score['macro'] += F1score[class_list[ii]]
        auc['macro'] += auc[class_list[ii]]
    precision['macro'] /= 4
    recall['macro'] /= 4
    F1score['macro'] /= 4
    auc['macro'] /= 4
    
    print(f'total: tp:{tp[4]}, fp:{fp[4]}, tn:{tn[4]}, fn:{fn[4]}')
    print(f"macro:  P:{precision['macro']}, R:{recall['macro']}, F1:{F1score['macro']}, auc:{auc['macro']}")

    per_patient_res = dict()
    per_patient_res['precision'] = precision
    per_patient_res['recall'] = recall
    per_patient_res['acc'] = acc
    per_patient_res['F1score'] = F1score
    per_patient_res['auc'] = auc
    return per_patient_res



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


     # load data
    print('--------------------loading data--------------------')
    img_paths, targets, patient_ids, img_ids = get_data(split_info_path)
    validset = Xinguan(img_paths[2], targets[2], patient_ids[2], img_ids[2])

    model = torchvision.models.resnet50(num_classes=4)
    model = model.to(device)    # 转gpu
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoints))    

    per_img_res, per_patient_res = evaluate(model,validset)         
