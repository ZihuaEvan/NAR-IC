import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle

import json
import cv2
from PIL import Image


class CocoDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, 
        input_seq, 
        target_seq,
        att_feats_folder, 
        seq_per_img,
        max_feat_num
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        # self.image_ids = utils.load_lines(image_ids_path)
        # 此处image_ids_path为ids2path的映射dict
        with open(image_ids_path, 'r') as f:
            self.ids2path = json.load(f)           # dict {image_id: image_path}
            self.image_ids = list(self.ids2path.keys())  # list of str
        
        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None

        # 构建图像预处理单元

        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0,:])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None
         
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.ids2path[image_id]
        indices = np.array([index]).astype('int')

        if self.att_feats_folder is not None:
            att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['feat']
            att_feats = np.array(att_feats).astype('float32')
            att_feats = torch.tensor(att_feats)
        else:
            att_feats = torch.zeros(1, 1)
        
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
           att_feats = att_feats[:self.max_feat_num, :]

        if self.seq_len < 0:
            return indices, att_feats

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
           
        n = len(self.input_seq[image_id])   
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)                
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix,:]
            target_seq[sid + i] = self.target_seq[image_id][ix,:]
        return indices, input_seq, target_seq, att_feats