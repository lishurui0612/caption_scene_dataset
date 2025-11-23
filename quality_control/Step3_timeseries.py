# coding=gbk
import os
import imageio
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import io
from scipy.signal import detrend
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    root = '/public_bme/data/lishr/Cross_modal/Data'
    processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'
    script_root = '/public/home/lishr2022/Project/Cross-modal/experiment/stimulus/scripts'

    vertices = [149079 + 151166,
                135103 + 135723,
                155295 + 151303,
                141922 + 142796,
                141578 + 138836,
                146440 + 149139,
                145747 + 144531,
                129958 + 128115]

    length = 18

    # Matched Trial
    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        sub_data_root = os.path.join(processed_root, subject, 'Stimulus', 'rescaled')
        savedir = os.path.join(processed_root, subject, 'quality_control', subject + '_mean_ts.npy')

        if os.path.exists(savedir):
            continue

        mean_ts = np.zeros((vertices[sub_id], 18))
        count = 0

        FileList = sorted(os.listdir(sub_data_root))
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

            data_dir = os.path.join(sub_data_root, file)
            data = np.squeeze(io.loadmat(data_dir)['data'])
            mean_data = np.abs(np.mean(data, axis=1, keepdims=True))
            data = detrend(data)
            data = data / mean_data * 100

            for index, tr in enumerate(onset):
                if tr + length >= 150:
                    continue
                count += 1
                mean_ts = mean_ts + data[:, int(tr):int(tr)+length]

        mean_ts = mean_ts / count
        np.save(savedir, mean_ts)

    # Unatched Trial - Main experiment
    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        sub_data_root = os.path.join(processed_root, subject, 'Stimulus', 'rescaled')
        savedir = os.path.join(processed_root, subject, 'quality_control', subject + '_mean_ts_unmatch_main.npy')

        if os.path.exists(savedir):
            continue

        mean_ts = np.zeros((vertices[sub_id], 18))
        count = 0

        FileList = sorted(os.listdir(sub_data_root))
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
                if temp[2] == '0' and temp[3] == '1':
                    onset.append(int(float(temp[1])) / 2 - 1)

            data_dir = os.path.join(sub_data_root, file)
            data = np.squeeze(io.loadmat(data_dir)['data'])
            mean_data = np.abs(np.mean(data, axis=1, keepdims=True))
            data = detrend(data)
            data = data / mean_data * 100

            for index, tr in enumerate(onset):
                if tr + length >= 150:
                    continue
                count += 1
                mean_ts = mean_ts + data[:, int(tr):int(tr)+length]

        mean_ts = mean_ts / count
        np.save(savedir, mean_ts)

    # Unmatched trial
    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        sub_data_root = os.path.join(processed_root, subject, 'unmatch', 'min_process')
        savedir = os.path.join(processed_root, subject, 'quality_control', subject + '_mean_ts_unmatch.npy')

        if not os.path.exists(sub_data_root):
            continue

        if os.path.exists(savedir):
            continue

        mean_ts = np.zeros((vertices[sub_id], 18))
        count = 0

        FileList = sorted(os.listdir(sub_data_root))
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

            data_dir = os.path.join(sub_data_root, file)
            data = np.squeeze(io.loadmat(data_dir)['data'])
            mean_data = np.abs(np.mean(data, axis=1, keepdims=True))
            data = detrend(data)
            data = data / mean_data * 100

            for index, tr in enumerate(onset):
                if tr + length >= 156:
                    continue
                count += 1
                mean_ts = mean_ts + data[:, int(tr):int(tr)+length]

        mean_ts = mean_ts / count
        np.save(savedir, mean_ts)

