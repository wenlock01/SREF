# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# 通过加载预测的CAM（Class Activation Map）数据，并结合CRF（Conditional Random Field）推理生成语义分割的伪标签，保存为PNG文件。

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default=r'E:\E-images\weakly\inria\train/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='inria_puzzleCAM_resnet50_8_0.01_wd5e-4@train@scale=1.0', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.35, type=float)
parser.add_argument('--bg_threshold', default=0.35, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.experiment_name
    
    pred_dir = f'F:\FlipCAM-master-master/experiments/inria-predictions/{experiment_name}/'
    aff_dir = create_directory('F:\FlipCAM-master-master/experiments/inria-predictions/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name, args.fg_threshold, args.bg_threshold))

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU
    meta_dic = read_json(r"E:\pycharm\FlipCAM-master-master\voc12\VOC_2012.json")
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    #################################################################################################
    # Convert
    #################################################################################################
    eval_timer = Timer()
    
    length = len(dataset)
    for step, (ori_image, image_id, _, _) in enumerate(dataset):
        png_path = aff_dir + image_id + '.png'
        if os.path.isfile(png_path):
            continue

        # load
        image = np.asarray(ori_image)
        cam_dict = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()

        ori_h, ori_w, c = image.shape

        ###SEAM###
        # cams = cam_dict[0]  # 直接获取CAM数据
        # if cams.ndim == 2:
        #     # 如果是2D数组，扩展为3D数组 (1, H, W)
        #     cams_3d = np.expand_dims(cams, axis=0)
        # else:
        #     cams_3d = cams
        # # keys可能需要根据实际情况设置，如果这是二分类问题，可能是[0, 1]
        # keys = np.array([0, 1])  # 根据您的具体任务调整

        keys = cam_dict['keys']
        cams = cam_dict['hr_cam']
        # cams = cam_dict['high_res']

        # 1. find confident fg & bg
        fg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
        fg_cam = np.argmax(fg_cam, axis=0)
        fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]
        
        bg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
        bg_cam = np.argmax(bg_cam, axis=0)
        bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
        
        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0
        
        imageio.imwrite(png_path, conf.astype(np.uint8))
        
        sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
        sys.stdout.flush()
    print()
    