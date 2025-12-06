# coding=gbk
import os
import gc
import copy
import torch
import random
import logging
import nilearn
import argparse
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from tqdm import tqdm
from scipy import io, stats
import cn_clip.clip as clip
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers.norm import LayerNorm2d
from sklearn.model_selection import KFold
from timm.models.convnext import ConvNeXtBlock
from typing import Any, Dict, List, Optional, Tuple, Union
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection

from dataset import prepare_pair_data, Caption_Image_dataset, Unmatch_dataset, Predictive_coding_dataset
from model import EncodingModel, PredictiveEncodingModel, LayerPreferrenceEncodingModel
from utils import get_parameter_number, cosine_sheduler, ExponentialMovingAverage


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--Stimulus_index_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt')
    parser.add_argument('--Image_Caption_pairs_root', type=str, default='/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt')
    parser.add_argument('--processed_root', type=str, default='/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data')
    parser.add_argument('--data_type', type=str, default='beta_zscore')
    parser.add_argument('--image_root', type=str, default='/public_bme/data/lishr/COCO_CN/All_images_480')
    parser.add_argument('--CLIP_model_root', type=str, default='/public/home/lishr2022/Project/Cross-modal/encoding/cross_encoding/model_cache')
    parser.add_argument('--output_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/reconstruction/encoding/test_001')
    parser.add_argument('--vit_model_root', type=str, default=None)
    parser.add_argument('--bert_model_root', type=str, default=None)
    parser.add_argument('--bert_ckpt_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/demo/opt_cap_model.pth')
    parser.add_argument('--vit_ckpt_dir', type=str, default='/public/home/lishr2022/Project/Cross-modal/demo/opt_img_model.pth')
    parser.add_argument('--trial_type', type=str, default='match')

    parser.add_argument('--behavior_in', type=int, default=8)
    parser.add_argument('--behavior_hidden', type=int, default=16)
    parser.add_argument('--final_visual_emb_dim', type=int, default=64)
    parser.add_argument('--final_bert_emb_dim', type=int, default=1024)
    parser.add_argument('--encode_type', type=str, default='visual')
    parser.add_argument('--index', type=int, default=3)
    parser.add_argument('--condition', type=int, default=0)
    parser.add_argument('--roi_type', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--start_warmup_value', type=float, default=1e-5)
    parser.add_argument('--ema_interval', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--reg_weight', type=float, default=3e-5)
    parser.add_argument('--pearson_weight', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--subject', type=str)

    return parser


def Calculate_R2(Feature_extractor, model, val_dataloader, device, args):
    pearsonr = PearsonCorrCoef(num_outputs=args.num_vertices).to(device)
    model.eval()
    with torch.no_grad():
        inputs = []
        prediction = []
        for i, sample in enumerate(val_dataloader):
            for i in range(len(sample)):
                if type(sample[i]) == torch.Tensor:
                    sample[i] = sample[i].to(device, dtype=torch.float)

            if args.condition == 0:
                sample[4] = torch.zeros_like(sample[4]).to(device, dtype=torch.float)
                sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)

            if args.encode_type == 'visual':
                feature = Feature_extractor.VisualEncode(sample, 3)
                outputs = model(feature[str(l)])

                inputs.append(sample[3])
                prediction.append(outputs)
            elif args.encode_type == 'caption':
                feature = Feature_extractor.BertEncode(sample, 2)
                outputs = model(feature[str(l)])

                inputs.append(sample[2])
                prediction.append(outputs)

    inputs = torch.cat(inputs, dim=0)
    prediction = torch.cat(prediction, dim=0)

    pearson = pearsonr(prediction, inputs)
    return pearson


if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    # 根据subject初始化
    if args.subject == 'S1':
        args.num_vertices = 300245
        args.lh_vertices = 149079
        args.rh_vertices = 151166
    elif args.subject == 'S2':
        args.num_vertices = 270826
        args.lh_vertices = 135103
        args.rh_vertices = 135723
    elif args.subject == 'S3':
        args.num_vertices = 306598
        args.lh_vertices = 155295
        args.rh_vertices = 151303
    elif args.subject == 'S4':
        args.num_vertices = 284718
        args.lh_vertices = 141922
        args.rh_vertices = 142796
    elif args.subject == 'S5':
        args.num_vertices = 280414
        args.lh_vertices = 141578
        args.rh_vertices = 138836
    elif args.subject == 'S6':
        args.num_vertices = 295579
        args.lh_vertices = 146440
        args.rh_vertices = 149139
    elif args.subject == 'S7':
        args.num_vertices = 290278
        args.lh_vertices = 145747
        args.rh_vertices = 144531
    elif args.subject == 'S8':
        args.num_vertices = 258073
        args.lh_vertices = 129958
        args.rh_vertices = 128115

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handle = logging.FileHandler(os.path.join(args.output_dir, 'record.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    Feature_extractor = LayerPreferrenceEncodingModel(
        behavior_in=args.behavior_in,
        behavior_hidden=args.behavior_hidden,
        CLIP_model_root=args.CLIP_model_root,
        encode_type=args.encode_type,
        vit_model_root=args.vit_model_root,
        bert_model_root=args.bert_model_root
    )
    Feature_extractor = Feature_extractor.to(device, dtype=torch.float)
    Feature_extractor.requires_grad_(False)
    Feature_extractor.eval()
    if args.encode_type == 'visual':
        layers = [1, 2, 4, 8, 14, 20, 26, 32]
    else:
        layers = [1, 2, 4, 8, 12, 16, 20, 24]

    Stimulus_index, Stimulus_pairs, stim_dict, train_list, val_list = prepare_pair_data(args.subject, args.Image_Caption_pairs_root, args.Stimulus_index_root, args.processed_root, data_type=args.data_type)
    data_list = train_list + val_list
    random.shuffle(data_list)

    decay_parameter = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    layers_R2 = torch.zeros((len(layers), args.num_vertices)).to(device)

    for layer_index, l in enumerate(layers):
        R2 = torch.zeros((args.n_splits, args.num_vertices)).to(device)
        kf = KFold(n_splits=args.n_splits, shuffle=False)
        for fold, (train_index, val_index) in enumerate(kf.split(data_list)):
            Fold_R2 = torch.zeros(args.num_vertices).to(device)
            for decay in decay_parameter:
                torch.cuda.empty_cache()

                train_list = [data_list[i] for i in train_index]
                val_list = [data_list[i] for i in val_index]

                train_dataset = Caption_Image_dataset(train_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)
                val_dataset = Caption_Image_dataset(val_list, Stimulus_index, Stimulus_pairs, stim_dict, args.image_root, norm=False)

                train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
                val_dataloader = data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

                if args.encode_type == 'visual':
                    model = nn.Sequential(
                        nn.Linear(1280, 50),
                        nn.Linear(50, args.num_vertices)
                    )
                elif args.encode_type == 'caption':
                    model = nn.Sequential(
                        nn.Linear(1024, 50),
                        nn.Linear(50, args.num_vertices)
                    )

                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

                model = model.to(device, dtype=torch.float)

                logger.info('---------------Start Training for Layer %d, Fold %d, weight decay %f---------------' % (l, fold, decay))
                for epoch in range(args.epochs):
                    for index, sample in enumerate(train_dataloader):
                        for i in range(len(sample)):
                            if type(sample[i]) == torch.Tensor:
                                sample[i] = sample[i].to(device, dtype=torch.float)

                        if args.condition == 0:
                            sample[4] = torch.zeros_like(sample[4]).to(device, dtype=torch.float)
                            sample[5] = torch.zeros_like(sample[5]).to(device, dtype=torch.float)

                        optimizer.zero_grad()
                        if args.encode_type == 'visual':
                            feature = Feature_extractor.VisualEncode(sample, 3)
                            outputs = model(feature[str(l)])
                            loss = F.mse_loss(outputs, sample[3])
                        elif args.encode_type == 'caption':
                            feature = Feature_extractor.BertEncode(sample, 2)
                            outputs = model(feature[str(l)])
                            loss = F.mse_loss(outputs, sample[2])

                        loss.backward()
                        optimizer.step()

                        if index % 100 == 0:
                            logger.info(
                                '[Epoch %d/%d] [Step %d/%d] [MSE_Loss: %f] [lr: %f]'
                                % (
                                    epoch, args.epochs, index, len(train_dataloader), loss.item(), optimizer.param_groups[0]['lr'])
                            )

                    # Validation
                    temp = Calculate_R2(Feature_extractor, model, val_dataloader, device, args)
                    temp = torch.square(temp)
                    Fold_R2 = torch.where(Fold_R2 > temp, Fold_R2, temp)
                    logger.info('Successfully update R2 for all vertices! Average R2: %.5f' % (torch.mean(Fold_R2).item()))

                    scheduler.step(torch.mean(temp))

            R2[fold, :] = Fold_R2
        result_R2 = torch.mean(R2, dim=0)
        layers_R2[layer_index, :] = result_R2

    layers_R2 = layers_R2.detach().cpu().numpy()
    savedir = os.path.join(args.output_dir, 'layer_preference.npy')
    np.save(savedir, layers_R2)

    logger.info('Successfully save R2 for all vertices!')

    logger.info('---------------Finish training---------------')