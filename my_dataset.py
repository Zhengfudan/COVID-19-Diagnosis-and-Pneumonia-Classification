# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:03:31 2019
@author: Chen

modified on 2020/3/8
@author: Wu

modified on 2020/3/17
@author: Wu

modified on 2020/3/18
@author: Wu

modified on 2020/3/28
@author Wu
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import csv
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

import random
from glob import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import collections
from collections import defaultdict
import json

class Xinguan(Dataset):
    def __init__(self, img_path_list, target_list, patient_ids, img_ids):
        super(Xinguan, self).__init__()
        self.img_path_list = img_path_list
        self.target_list = target_list
        self.patient_ids = patient_ids
        self.img_ids = img_ids
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        target = self.target_list[index]
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img = self.preprocess(img)
        if self.patient_ids is None:
            return {'image': torch.from_numpy(img).float(), 'target': torch.Tensor(target)}
        else:
            patient_id = self.patient_ids[index]
            img_id = self.img_ids[index]
            return {'image': torch.from_numpy(img).float(), 'target': target, 
                    'patient_id': patient_id, 'img_id': img_id}

    def __len__(self): 
        return len(self.target_list)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

def resplit(split_info_path):
    # 读原先数据集
    split_info_paths = []
    split_info_paths.append(os.path.join(split_info_path, 'train_augumented.json'))
    split_info_paths.append(os.path.join(split_info_path, 'val.json'))
    split_info_paths.append(os.path.join(split_info_path, 'test.json')) 
    img_paths = []
    targets = []  
    patient_ids = []
    img_ids = []
    for i in range(3): 
        with open(split_info_paths[i], 'r') as json_file:
            dict_dataset = json.load(json_file)
            img_paths += dict_dataset['img_paths']
            targets += dict_dataset['targets']
            patient_ids += dict_dataset['patient_ids']
            img_ids += dict_dataset['img_ids']
    # 根据病人级划分数据集
    train_idx = []
    valid_idx = []
    test_idx = []
    patient_ids_of_tvt = [set(), set(), set(), set()]   # 四类
    for pid, tg in zip(patient_ids, targets):
        patient_ids_of_tvt[tg].add(pid)
    for i in range(4):
        idx = list(patient_ids_of_tvt[i])
        train_idx_sub, test_idx_sub = train_test_split(idx, test_size=0.4)
        valid_idx_sub, test_idx_sub = train_test_split(test_idx_sub, test_size=30)
        train_idx += train_idx_sub
        valid_idx += valid_idx_sub
        test_idx += test_idx_sub
    # 构造patient id的字典
    train_val_test_dict = dict()
    for i in train_idx:
        train_val_test_dict[i] = 0
    for i in valid_idx:
        train_val_test_dict[i] = 1
    for i in test_idx:
        train_val_test_dict[i] = 2
    # 重新划分数据集
    mydicts = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for i, patient_id in enumerate(patient_ids):
        datatype = train_val_test_dict[patient_id]
        mydicts[datatype]['img_paths'].append(img_paths[i])
        mydicts[datatype]['patient_ids'].append(patient_ids[i])
        mydicts[datatype]['img_ids'].append(img_ids[i])
        mydicts[datatype]['targets'].append(targets[i])
    # 写入磁盘   
    output_files = []
    output_files.append(os.path.join(split_info_path, 'train_631_new.json'))
    output_files.append(os.path.join(split_info_path, 'val_631_new.json'))
    output_files.append(os.path.join(split_info_path, 'test_631_new.json'))
    for i in range(3):
        with open(output_files[i], 'w') as output_file:
            json.dump(mydicts[i], output_file)
def get_data(split_info_path):
    if not os.path.exists(os.path.join(split_info_path, 'train_631_new.json')):
        resplit(split_info_path)

    split_info_paths = []
    split_info_paths.append(os.path.join(split_info_path, 'train_631_new.json'))
    split_info_paths.append(os.path.join(split_info_path, 'val_631_new.json'))
    split_info_paths.append(os.path.join(split_info_path, 'test_631_new.json')) 
    img_paths = [[],[],[]]
    targets = [[],[],[]]  
    patient_ids = [[],[],[]]
    img_ids = [[],[],[]]
    for i in range(3): 
        with open(split_info_paths[i], 'r') as json_file:
            dict_dataset = json.load(json_file)
            img_paths[i] = dict_dataset['img_paths']
            targets[i] = dict_dataset['targets']
            patient_ids[i] = dict_dataset['patient_ids']
            img_ids[i] = dict_dataset['img_ids']
    return img_paths, targets, patient_ids, img_ids
    

def split_data(output_folder):
    # 4 kinds of dataset. if you need to train your own dataset. plz change the folders here.
    normal_folders = []
    xinguan_folders = []
    xijun_folders =  []
    bingdu_folders = []
    folder_list = [normal_folders, xinguan_folders, xijun_folders, bingdu_folders]
    patient_ids = [[],[],[],[]]
    img_ids = [[],[],[],[]]
    img_paths = [[],[],[],[]]
    start_of_p = start_of_i = 0
    for i in range(4):
        img_paths[i], patient_ids[i], img_ids[i], start_of_p, start_of_i = get_splited_path_list(folder_list[i], start_of_p, start_of_i)
    mydicts = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for i in range(3):  # iter on train valid test
        for j in range(4):  # iter on 4 classes
            mydicts[i]['img_paths'] += img_paths[j][i]
            mydicts[i]['patient_ids'] +=  patient_ids[j][i]
            mydicts[i]['img_ids'] += img_ids[j][i]
            mydicts[i]['targets'] += (np.ones(len(img_ids[j][i]), dtype=int) * j).tolist()
    output_files = []
    output_files.append(os.path.join(output_folder, 'train.json'))
    output_files.append(os.path.join(output_folder, 'val.json'))
    output_files.append(os.path.join(output_folder, 'test.json'))
    for i in range(3):
        with open(output_files[i], 'w') as output_file:
            json.dump(mydicts[i], output_file)


def get_splited_path_list(folder_list, start_of_p=0, start_of_i=0):
    # get patient notes
    patient_note_set = set()
    for folder in folder_list[0]:
        file_names = os.listdir(folder)
        for file_name in file_names:
            patient_note = file_name.split('_')[0]
            patient_note_set.add(patient_note)

    # get patient ids
    patient_id = start_of_p
    patient_id_dict = dict()
    for patient_note in patient_note_set:
        patient_id_dict[patient_note] = patient_id
        patient_id += 1

    # split patient ids
    id_list = [i+start_of_p for i in range(patient_id-start_of_p)]
    train_ids, val_ids = train_test_split(id_list, test_size=1/9)

    patient_note_set = set()
    for folder in folder_list[1]:
        file_names = os.listdir(folder)
        for file_name in file_names:
            patient_note = file_name.split('_')[0]
            patient_note_set.add(patient_note)
    start_of_p = patient_id
    for patient_note in patient_note_set:
        patient_id_dict[patient_note] = patient_id
        patient_id += 1

    id_list2 = [i+start_of_p for i in range(patient_id-start_of_p)]
    start_of_p = patient_id
    train_ids2, test_ids = train_test_split(id_list2, test_size=30)
    train_ids2, val_ids2 = train_test_split(train_ids2, test_size=1/9)

    id_list += id_list2
    train_ids += train_ids2
    val_ids += val_ids2

    train_val_test_dict = dict()
    for i in train_ids:
        train_val_test_dict[i] = 0
    for i in val_ids:
        train_val_test_dict[i] = 1
    for i in test_ids:
        train_val_test_dict[i] = 2

    # split img path
    splited_paths = [[], [], []] # 0 for train, 1 for valid, 2 for test
    img_ids = [[], [], []]
    patient_ids = [[],[],[]]
    for i in range(2):
        for folder in folder_list[i]:
            file_names = os.listdir(folder)
            for file_name in file_names:
                patient_note = file_name.split('_')[0]
                patient_id = patient_id_dict[patient_note]
                train_val_test = train_val_test_dict[patient_id]
                splited_paths[train_val_test].append(os.path.join(folder, file_name))
                img_ids[train_val_test].append(start_of_i + 0.0)
                start_of_i += 1
                patient_ids[train_val_test].append(patient_id + 0.0)

                if train_val_test == 0:
                    additional_file = folder.split('/')
                    additional_file[-1] = 'filled'
                    additional_file = '/'.join(additional_file)
                    splited_paths[0].append(os.path.join(additional_file, file_name))
                    patient_ids[0].append(patient_id)
                    img_ids[0].append(start_of_i)
                    start_of_i += 1

    return splited_paths, patient_ids, img_ids, start_of_p, start_of_i
