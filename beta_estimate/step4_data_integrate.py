import os
import torch
import random
import nilearn
import numpy as np
from scipy import io
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
import matplotlib.pyplot as plt
from cn_clip.clip import load_from_name, image_transform

if __name__ == '__main__':
    script_root = '/public/home/lishr2022/Project/Cross-modal/experiment/stimulus/scripts'
    stimulus_index_root = '/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt'

    data_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Data'
    processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'

    Stimulus_index = dict()
    with open(stimulus_index_root, 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_index[int(temp[0])] = temp[1]
            Stimulus_index[temp[1]] = int(temp[0])
    f.close()

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    for index, subject in enumerate(subjects):
        print('For %s' % subject)
        subject_script_root = os.path.join(script_root, 'subject' + subject[1])
        subject_behavior_root = os.path.join(data_root, subject, 'behavior', 'stimulus')

        subject_beta_root = os.path.join(processed_root, subject, 'Stimulus', 'beta')
        behavior_list = sorted(os.listdir(subject_behavior_root))

        if not os.path.exists(subject_beta_root) or len(os.listdir(subject_beta_root)) != 8800:
            continue

        for run in tqdm(range(1, 201)):
            if run <= 50 or (100 < run and run <= 150):
                repeat = 1
            else:
                repeat = 2
            mat = None
            for file in behavior_list:
                if 'run' + str(run) + '_' in file and '.mat' in file:
                    mat = io.loadmat(os.path.join(subject_behavior_root, file))
                    behavior_response = mat['theData'][0]['resp'][0][0]
                    behavior_rt = mat['theData'][0]['rt'][0][0]
                    trial_unmatch = mat['theSubject'][0]['trial'][0]['Unmatch'][0][0][0]
            with open(os.path.join(subject_script_root, 'subject' + subject[1] + '_run' + str(run) + '.txt'), 'r', encoding='gbk') as f:
                run_stim = f.readlines()
            f.close()

            timestamp = 0
            flag = dict()
            for line in run_stim:
                temp = line.split()
                if temp[0] == 'Index':
                    continue
                if temp[4] == 'None':
                    timestamp += 2
                    continue
                timestamp += 1
                caption_stim = Stimulus_index[temp[5]]
                if caption_stim not in flag:
                    flag[caption_stim] = 1
                    beta_dir = os.path.join(subject_beta_root, subject + '_stim_' + str(caption_stim).zfill(5) + '_Run_' + str(run).zfill(3) + '_beta.mat')
                else:
                    flag[caption_stim] += 1
                    beta_dir = os.path.join(subject_beta_root, subject + '_stim_' + str(caption_stim).zfill(5) + '_Run_' + str(run).zfill(3) + '_beta_' + str(flag[caption_stim]) + '.mat')
                # beta = np.squeeze(io.loadmat(beta_dir)['beta'])
                beta = np.squeeze(io.loadmat(beta_dir)['beta_data'])
                pair_stim = np.squeeze(io.loadmat(beta_dir)['pair_stim'])
                io.savemat(beta_dir,
                           {'beta': beta, 'run': run, 'timestamp': timestamp, 'behavior': -1, 'behavior_rt': -1,
                            'caption': 1, 'image': 0, 'unmatch': -1, 'repeat': repeat, 'pair_stim': pair_stim, 'pair_unmatch': trial_unmatch[(timestamp+1) // 2 - 1]})

                timestamp += 1
                image_stim = Stimulus_index[temp[4]]
                if image_stim not in flag:
                    flag[image_stim] = 1
                    beta_dir = os.path.join(subject_beta_root, subject + '_stim_' + str(image_stim).zfill(5) + '_Run_' + str(run).zfill(3) + '_beta.mat')
                else:
                    flag[image_stim] += 1
                    beta_dir = os.path.join(subject_beta_root,
                                            subject + '_stim_' + str(image_stim).zfill(5) + '_Run_' + str(
                                                run).zfill(3) + '_beta_' + str(flag[image_stim]) + '.mat')
                # beta = np.squeeze(io.loadmat(beta_dir)['beta'])
                beta = np.squeeze(io.loadmat(beta_dir)['beta_data'])
                pair_stim = np.squeeze(io.loadmat(beta_dir)['pair_stim'])
                if mat != None:
                    io.savemat(beta_dir, {'beta': beta, 'run': run, 'timestamp': timestamp,
                                          'behavior': behavior_response[timestamp // 2 - 1],
                                          'behavior_rt': behavior_rt[timestamp // 2 - 1], 'caption': 0, 'image': 1,
                                          'unmatch': trial_unmatch[timestamp // 2 - 1], 'repeat': repeat, 'pair_stim': pair_stim})
                else:
                    io.savemat(beta_dir, {'beta': beta, 'run': run, 'timestamp': timestamp,
                                          'behavior': -1, 'behavior_rt': -1, 'caption': 0, 'image': 1,
                                          'unmatch': -1, 'repeat': repeat, 'pair_stim': pair_stim})
                        