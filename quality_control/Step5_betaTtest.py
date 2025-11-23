# coding=gbk
import os
import re
import h5py
import shutil
import numpy as np
import pandas as pd
from scipy import io
import nibabel as nib
from tqdm import tqdm
import scipy.stats as stats
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    root = '/public_bme/data/lishr/Cross_modal/Data'
    processed_root = '/public_bme/data/lishr/Cross_modal/Processed_Data'
    script_root = '/public/home/lishr2022/Project/Cross-modal/experiment/stimulus/scripts'

    vertices = [149079 + 151166,
                135103 + 135723,
                155295 + 151303,
                141922 + 142796,
                141578 + 138836,
                146440 + 149139,
                145747 + 144531,
                129958 + 128115]

    hemi_vertices = [
        [149079, 151166],
        [135103, 135723],
        [155295, 151303],
        [141922, 142796],
        [141578, 138836],
        [146440, 149139],
        [145747, 144531],
        [129958, 128115]
    ]

    # Image & Caption correlation
    Stimulus_index = {}
    with open('/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt', 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_index[temp[1]] = int(temp[0])
            Stimulus_index[int(temp[0])] = temp[1]
    f.close()

    Caption_Image_pairs = {}
    with open('/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt', 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Caption_Image_pairs[temp[0]] = temp[1]
            Caption_Image_pairs[temp[1]] = temp[0]
    f.close()

    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        match_data_dir = os.path.join(processed_root, subject, 'Stimulus', 'beta_zscore')
        unmatch_data_dir = os.path.join(processed_root, subject, 'unmatch', 'beta')

        if not os.path.exists(match_data_dir) or not os.path.exists(unmatch_data_dir):
            continue

        match_data = []
        FileList = sorted(os.listdir(match_data_dir))
        for file in tqdm(FileList):
            if file[-5] != 'a':
                stim = int(file[-24:-19])
            else:
                stim = int(file[-22:-17])

            if Stimulus_index[stim][-3:] == 'jpg':
                continue

            dir = os.path.join(match_data_dir, file)
            data = io.loadmat(dir)
            beta = np.squeeze(data['beta'])
            pair_unmatch = data['pair_unmatch'][0][0]

            if pair_unmatch == 0:
                match_data.append(beta)
        match_data = np.array(match_data)

        unmatch_data = []
        FileList = sorted(os.listdir(unmatch_data_dir))
        for file in tqdm(FileList):
            if file[-5] != 'a':
                stim = int(file[-24:-19])
            else:
                stim = int(file[-22:-17])

            if Stimulus_index[stim][-3:] == 'jpg':
                continue

            dir = os.path.join(unmatch_data_dir, file)
            data = io.loadmat(dir)
            beta = np.squeeze(data['beta'])
            pair_stim = int(np.squeeze(data['pair_stim']))

            if Stimulus_index[Caption_Image_pairs[Stimulus_index[pair_stim]]] != stim:
                unmatch_data.append(beta)
        unmatch_data = np.array(unmatch_data)

        t_value = np.zeros(vertices[sub_id])
        p_value = np.zeros(vertices[sub_id])
        for vertice in tqdm(range(vertices[sub_id])):
            t_value[vertice], p_value[vertice] = stats.ttest_ind(match_data[:, vertice], unmatch_data[:, vertice])

        lh = {
            't_value': t_value[:hemi_vertices[sub_id][0]],
            'p_value': p_value[:hemi_vertices[sub_id][0]]
        }
        lh_df = pd.DataFrame(lh)
        savedir = os.path.join(processed_root, subject, 'quality_control', subject + '_match_unmatch_beta_lh.csv')
        lh_df.to_csv(savedir, index=False)

        rh = {
            't_value': t_value[hemi_vertices[sub_id][0]:],
            'p_value': p_value[hemi_vertices[sub_id][0]:]
        }
        rh_df = pd.DataFrame(rh)
        savedir = os.path.join(processed_root, subject, 'quality_control', subject + '_match_unmatch_beta_rh.csv')
        rh_df.to_csv(savedir, index=False)

