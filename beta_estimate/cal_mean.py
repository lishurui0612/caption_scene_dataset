# coding=gbk
import os
import gc
import torch
import random
import logging
import nilearn
import argparse
import numpy as np
import pandas as pd
from torch import nn
from scipy import io
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from einops import rearrange
from nilearn import plotting
import torch.utils.data as data
import matplotlib.pyplot as plt
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from einops import rearrange, repeat
from info_nce import InfoNCE, info_nce
from scipy.interpolate import griddata
from timm.layers.norm import LayerNorm2d
from einops.layers.torch import Rearrange
from timm.models.convnext import ConvNeXtBlock
from sklearn.metrics import calinski_harabasz_score
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from cn_clip.clip import load_from_name, image_transform, load_from_name, _tokenizer
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score, MetricCollection


def plot_weight(stat, subject, hemi_vertices, savedir, norm=True):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        axes.append(ax)

    vmax = 1
    if norm:
        stat = stat / vmax
        vmax = 1

    view = ['lateral', 'ventral', 'medial', 'dorsal']

    left_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'lh.inflated')
    right_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'rh.inflated')

    left_stat = stat[:hemi_vertices[0]]
    right_stat = stat[hemi_vertices[0]:]
    # plot left hemisphere
    print('- - - plotting left hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh=left_surf_mesh,
            stat_map=left_stat,
            hemi='left',
            view=view[i],
            vmax=vmax,
            axes=axes[i],
            title='Left - ' + view[i]
        )
    # plot right hemisphere
    print('- - - plotting right hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh=right_surf_mesh,
            stat_map=right_stat,
            hemi='right',
            view=view[i],
            vmax=vmax,
            axes=axes[i + 4],
            title='Right - ' + view[i]
        )

    fig.savefig(savedir)


def process_file(Stimulus_index, file_path, image_mean, caption_mean):
    beta = np.squeeze(io.loadmat(file_path)['beta'])
    if file_path[-5] != 'a':
        stim = int(file_path[-24:-19])
    else:
        stim = int(file_path[-22:-17])
    if Stimulus_index[stim][-3:] == 'jpg':
        image_mean += beta
    else:
        caption_mean += beta
    return image_mean, caption_mean


def process_subject(Stimulus_index, subject, processed_root, num_vertices):
    subject_root = os.path.join(processed_root, subject, 'Stimulus', 'beta_zscore')
    caption_mean = np.zeros(num_vertices[subject])
    image_mean = np.zeros(num_vertices[subject])
    file_list = os.listdir(subject_root)

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, Stimulus_index, os.path.join(subject_root, file), image_mean, caption_mean): file for file in file_list}

        for future in as_completed(future_to_file):
            try:
                image_mean, caption_mean = future.result()
            except Exception as exc:
                print(f'File generated an exception: {exc}')

    num_files = len(file_list)
    caption_mean /= num_files
    image_mean /= num_files

    print(caption_mean)
    print(image_mean)

    caption_percentiles = np.percentile(np.abs(caption_mean), range(0, 101, 1))
    image_percentiles = np.percentile(np.abs(image_mean), range(0, 101, 1))
    print(caption_percentiles)
    print(image_percentiles)


if __name__ == '__main__':
    processed_root = '/public_bme/data/lishr/Cross_modal/Processed_Data'
    subjects = ['S1_LiJiawei']

    num_vertices = {
        'S1_LiJiawei': 300245,
        'S2_ChenPeili': 270826,
        'S3_WangZhenjie': 306598,
        'S4_WangRuoming': 0,
        'S5_FangYu': 0,
        'S6_WuJinze': 0,
        'S7_LiuAnglin': 0,
        'S8_ChenQian': 0
    }

    hemi_vertices = {
        'S1_LiJiawei': [149079, 151166]
    }

    Stimulus_index_root = '/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt'
    Stimulus_index = dict()
    with open(Stimulus_index_root, 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_index[int(temp[0])] = temp[1]
            Stimulus_index[temp[1]] = int(temp[0])
    f.close()

    # for subject in subjects:
    #     process_subject(Stimulus_index, subject, processed_root, num_vertices)

    for subject in subjects:
        subject_root = os.path.join(processed_root, subject, 'Stimulus', 'beta_zscore')
        file_list = os.listdir(subject_root)
        caption = []
        image = []
        flag = []
        for file in tqdm(file_list):
            if file in flag:
                continue
            if file[-5] != 'a':
                stim = int(file[-24:-19])
            else:
                stim = int(file[-22:-17])
            data = io.loadmat(os.path.join(subject_root, file))

            pair_stim = data['pair_stim'][0][0]
            beta = data['beta']
            flag.append(file)

            if Stimulus_index[stim][-3:] == 'jpg':
                image.append(beta)
                if file[-5] != 'a':
                    pair_file = file[:-24] + str(pair_stim).zfill(5) + file[-19:]
                else:
                    pair_file = file[:-22] + str(pair_stim).zfill(5) + file[-17:]
                temp = io.loadmat(os.path.join(subject_root, pair_file))['beta']
                caption.append(temp)
        image = np.concatenate(image, axis=0)
        caption = np.concatenate(caption, axis=0)

        corr = np.zeros(num_vertices[subject])
        p = np.zeros(num_vertices[subject])
        for v in tqdm(range(num_vertices[subject])):
            corr[v], p[v] = pearsonr(caption[:, v], image[:, v])
        savedir = os.path.join(processed_root, subject, 'Stimulus', 'corr.png')
        plot_weight(corr, subject, hemi_vertices[subject], savedir, norm=False)

        lh = {
            'correlation': corr[:hemi_vertices[subject][0]],
            'p-value': p[:hemi_vertices[subject][0]]
        }
        savedir = os.path.join(processed_root, subject, 'Stimulus', 'lh_corr.csv')
        df = pd.DataFrame(lh)
        df.to_csv(savedir, index=False)

        rh = {
            'correlation': corr[hemi_vertices[subject][0]:],
            'p-value': p[hemi_vertices[subject][0]:]
        }
        savedir = os.path.join(processed_root, subject, 'Stimulus', 'rh_corr.csv')
        df = pd.DataFrame(rh)
        df.to_csv(savedir, index=False)
