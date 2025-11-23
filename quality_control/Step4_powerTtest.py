# coding=gbk
import os
import imageio
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.signal import detrend
from scipy import io, stats
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


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

    length = 18

    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        match_data_dir = os.path.join(processed_root, subject, 'Stimulus', 'rescaled')
        unmatch_data_dir = os.path.join(processed_root, subject, 'unmatch', 'min_process')

        if not os.path.exists(match_data_dir) or not os.path.exists(unmatch_data_dir):
            continue

        # Match trial power
        match_data = []
        FileList = sorted(os.listdir(match_data_dir))
        for run, file in tqdm(enumerate(FileList), total=len(FileList)):
            script_dir = os.path.join(sub_script_root, 'subject'+str(sub_id+1)+'_run'+str(run+1)+'.txt')
            with open(script_dir, 'r', encoding='gbk') as f:
                content = f.readlines()
            f.close()

            onset = []
            for index, line in enumerate(content):
                if index == 0:
                    continue
                temp = line.split()
                if temp[2] == '0' and temp[3] == '0':
                    onset.append(int(float(temp[1])) / 2 - 1)

            data_dir = os.path.join(match_data_dir, file)
            data = np.squeeze(io.loadmat(data_dir)['data'])
            data = (data - np.mean(data, axis=1, keepdims=True)) / np.mean(data, axis=1, keepdims=True) * 100
            data = detrend(data)

            for index, tr in enumerate(onset):
                if tr + length >= 150:
                    continue
                temp = data[:, int(tr):int(tr)+length]
                sequence_power = np.var(temp, axis=1)
                match_data.append(sequence_power)
        match_data = np.array(match_data)

        # Unmatch trial power
        unmatch_data = []
        FileList = sorted(os.listdir(unmatch_data_dir))
        for run, file in tqdm(enumerate(FileList), total=len(FileList)):
            script_dir = os.path.join(sub_script_root, 'subject'+str(sub_id+1)+'_run'+str(run+375)+'.txt')
            with open(script_dir, 'r', encoding='gbk') as f:
                content = f.readlines()
            f.close()

            onset = []
            for index, line in enumerate(content):
                if index == 0:
                    continue
                temp = line.split()
                if temp[3] == '1':
                    onset.append(int(float(temp[1])) / 2 - 1)

            data_dir = os.path.join(unmatch_data_dir, file)
            data = np.squeeze(io.loadmat(data_dir)['data'])
            data = (data - np.mean(data, axis=1, keepdims=True)) / np.mean(data, axis=1, keepdims=True) * 100
            data = detrend(data)

            for index, tr in enumerate(onset):
                if tr + length >= 156:
                    continue
                temp = data[:, int(tr):int(tr)+length]
                sequence_power = np.var(temp, axis=1)
                unmatch_data.append(sequence_power)
        unmatch_data = np.array(unmatch_data)

        # Test suppression
        mean_match = np.mean(match_data, axis=0)
        mean_unmatch = np.mean(unmatch_data, axis=0)
        diff_rate = (mean_unmatch - mean_match) / mean_match
        t_value = np.zeros(vertices[sub_id])
        p_value = np.zeros(vertices[sub_id])

        for i in tqdm(range(vertices[sub_id])):
            t_value[i], p_value[i] = stats.ttest_ind(match_data[:, i], unmatch_data[:, i])

        lh = {
            't_value': t_value[:hemi_vertices[sub_id][0]],
            'p_value': p_value[:hemi_vertices[sub_id][0]],
            'mean_match': mean_match[:hemi_vertices[sub_id][0]],
            'mean_unmatch': mean_unmatch[:hemi_vertices[sub_id][0]],
            'diff_rate': diff_rate[:hemi_vertices[sub_id][0]]
        }
        lh_df = pd.DataFrame(lh)
        savedir = os.path.join(processed_root, subject, 'quality_control', subject+'_match_unmatch_lh.csv')
        lh_df.to_csv(savedir, index=False)

        rh = {
            't_value': t_value[hemi_vertices[sub_id][0]:],
            'p_value': p_value[hemi_vertices[sub_id][0]:],
            'mean_match': mean_match[hemi_vertices[sub_id][0]:],
            'mean_unmatch': mean_unmatch[hemi_vertices[sub_id][0]:],
            'diff_rate': diff_rate[hemi_vertices[sub_id][0]:]
        }
        rh_df = pd.DataFrame(rh)
        savedir = os.path.join(processed_root, subject, 'quality_control', subject+'_match_unmatch_rh.csv')
        rh_df.to_csv(savedir, index=False)
