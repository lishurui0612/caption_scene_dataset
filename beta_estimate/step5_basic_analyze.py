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


def plot_weight(stat, subject, hemi_vertices, savedir, vmax=1.0):
    fig = plt.figure(figsize=(24, 8))
    axes = []
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        axes.append(ax)

    # vmax = np.max(abs(stat))
    # if vmax != 0.8:
    #     stat = stat / vmax

    view = ['lateral', 'ventral', 'medial', 'dorsal']

    left_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'lh.inflated')
    right_surf_mesh = os.path.join('/public_bme/data/lishr/Cross_modal/subjects', subject[3:], 'surf', 'rh.inflated')

    left_stat = stat[:hemi_vertices[0]]
    right_stat = stat[hemi_vertices[0]:]
    # plot left hemisphere
    print('- - - plotting left hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh = left_surf_mesh,
            stat_map = left_stat,
            hemi = 'left',
            view = view[i],
            vmax = vmax,
            axes = axes[i],
            title = 'Left - ' + view[i]
        )
    # plot right hemisphere
    print('- - - plotting right hemisphere')
    for i in range(4):
        plotting.plot_surf_stat_map(
            surf_mesh = right_surf_mesh,
            stat_map = right_stat,
            hemi = 'right',
            view = view[i],
            vmax = vmax,
            axes = axes[i+4],
            title = 'Right - ' + view[i]
        )

    fig.savefig(savedir)
    plt.close(fig)


if __name__ == '__main__':
    num_vertices = {
        'S1': 300245,
        'S2': 270826,
        'S3': 306598,
        'S4': 284718,
        'S5': 280414,
        'S6': 295579,
        'S7': 290278,
        'S8': 258073
    }

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

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    script_root = '/public/home/lishr2022/Project/Cross-modal/experiment/stimulus/scripts'

    data_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Data'
    beta_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'

    Caption_Stimulus = []
    Image_Stimulus = []
    with open('/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt', 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            if temp[1][-3:] == 'jpg':
                Image_Stimulus.append(int(temp[0]))
            else:
                Caption_Stimulus.append(int(temp[0]))
    f.close()

    # Noise ceiling
    for index, subject in enumerate(subjects):
        print('For %s' % subject)

        savedir = os.path.join(beta_root, subject, 'Stimulus', 'basic_analysis')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        subject_beta_root = os.path.join(beta_root, subject, 'Stimulus/beta_zscore')

        if not os.path.exists(subject_beta_root) or len(os.listdir(subject_beta_root)) != 8800:
            continue

        lh_caption_target = os.path.join(savedir, subject + '_lh_cap_nc.csv')
        rh_caption_target = os.path.join(savedir, subject + '_rh_cap_nc.csv')

        lh_img_target = os.path.join(savedir, subject + '_lh_img_nc.csv')
        rh_img_target = os.path.join(savedir, subject + '_rh_img_nc.csv')

        count = np.zeros(20000)
        FileList = os.listdir(subject_beta_root)
        for file in FileList:
            if file[-5] == '2':
                count[int(file[-24:-19])] += 1
            else:
                count[int(file[-22:-17])] += 1

        noise = []
        data = []
        for i in tqdm(range(20000)):
            if not i in Caption_Stimulus:
                continue
            temp = np.zeros((int(count[i]), num_vertices[subject]))
            cnt = -1
            if count[i] == 2:
                for file in FileList:
                    if subject + '_stim_' + str(i).zfill(5) in file:
                        cnt += 1
                        temp[cnt] = np.squeeze(io.loadmat(os.path.join(subject_beta_root, file))['beta'])
                        data.append(np.squeeze(io.loadmat(os.path.join(subject_beta_root, file))['beta']))
                # calculate variance
                stim_noise = np.var(temp, axis=0, ddof=1)
                noise.append(stim_noise)
        # average the variance and calculate the square root
        sigma_noise = np.sqrt(np.mean(noise, axis=0))

        # calculate the sigma_signal
        # sigma_signal = 1 - np.square(sigma_noise)
        sigma_signal = np.var(data, axis=0, ddof=1) - np.square(sigma_noise)
        sigma_signal[sigma_signal < 0] = 0
        sigma_signal = np.sqrt(sigma_signal)

        ncsnr = sigma_signal / sigma_noise
        NC = 100 * np.square(ncsnr) / (np.square(ncsnr) + 1 / 2)

        lh = {
            'Noise ceiling': NC[:hemi_vertices[index][0]]
        }
        rh = {
            'Noise ceiling': NC[hemi_vertices[index][0]:]
        }

        lh_NC = pd.DataFrame(lh)
        rh_NC = pd.DataFrame(rh)

        lh_NC.to_csv(lh_caption_target, index=False)
        rh_NC.to_csv(rh_caption_target, index=False)

        fig_dir = os.path.join(savedir, 'caption_nc.png')
        plot_weight(NC, subject, hemi_vertices[index], fig_dir, vmax=70)

        noise = []
        data = []
        for i in tqdm(range(20000)):
            if not i in Image_Stimulus:
                continue
            temp = np.zeros((int(count[i]), num_vertices[subject]))
            cnt = -1
            if count[i] == 2:
                for file in FileList:
                    if subject + '_stim_' + str(i).zfill(5) in file:
                        cnt += 1
                        temp[cnt] = np.squeeze(io.loadmat(os.path.join(subject_beta_root, file))['beta'])
                        data.append(np.squeeze(io.loadmat(os.path.join(subject_beta_root, file))['beta']))
                # calculate variance
                stim_noise = np.var(temp, axis=0, ddof=1)
                noise.append(stim_noise)
        # average the variance and calculate the square root
        sigma_noise = np.sqrt(np.mean(noise, axis=0))

        # calculate the sigma_signal
        # sigma_signal = 1 - np.square(sigma_noise)
        sigma_signal = np.var(data, axis=0, ddof=1) - np.square(sigma_noise)
        sigma_signal[sigma_signal < 0] = 0
        sigma_signal = np.sqrt(sigma_signal)

        ncsnr = sigma_signal / sigma_noise
        NC = 100 * np.square(ncsnr) / (np.square(ncsnr) + 1 / 2)

        lh = {
            'Noise ceiling': NC[:hemi_vertices[index][0]]
        }
        rh = {
            'Noise ceiling': NC[hemi_vertices[index][0]:]
        }

        lh_NC = pd.DataFrame(lh)
        rh_NC = pd.DataFrame(rh)

        lh_NC.to_csv(lh_img_target, index=False)
        rh_NC.to_csv(rh_img_target, index=False)

        fig_dir = os.path.join(savedir, 'image_nc.png')
        plot_weight(NC, subject, hemi_vertices[index], fig_dir, vmax=70)

    # Image & Caption correlation
    Stimulus_index = {}
    with open('/public/home/lishr2022/Project/Cross-modal/beta_estimate/Stimulus_index.txt', 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Stimulus_index[temp[1]] = int(temp[0])
            Stimulus_index[int(temp[0])] = temp[1]
    f.close()

    Caption_Image_pairs = []
    with open('/public_bme/data/lishr/COCO_CN/Selected_Match_Image_Caption_pairs.txt', 'r', encoding='gbk') as f:
        content = f.readlines()
        for line in content:
            temp = line.split()
            Caption_Image_pairs.append([temp[0], temp[1]])
    f.close()

    for i, subject in enumerate(subjects):
        print('For %s' % subject)

        subject_beta_root = os.path.join(beta_root, subject, 'Stimulus/beta_zscore')
        subject_result_root = os.path.join(beta_root, subject, 'Stimulus', 'basic_analysis')

        if not os.path.exists(subject_result_root):
            os.makedirs(subject_result_root)

        if not os.path.exists(subject_beta_root) or len(os.listdir(subject_beta_root)) != 8800:
            continue

        lh_savedir = os.path.join(subject_result_root, subject + '_cor_lh.csv')
        rh_savedir = os.path.join(subject_result_root, subject + '_cor_rh.csv')
        FileList = os.listdir(subject_beta_root)

        beta_caption = np.zeros((4400, num_vertices[subject]))
        beta_image = np.zeros((4400, num_vertices[subject]))

        count = 0
        for stim in tqdm(range(len(Caption_Image_pairs))):
            if not Caption_Image_pairs[stim][0] in Stimulus_index:
                continue
            image_pattern = r".*_stim_" + str(Stimulus_index[Caption_Image_pairs[stim][0]]).zfill(5) + '.*'
            caption_stim_index = Stimulus_index[Caption_Image_pairs[stim][1]]

            matched_files = [file for file in FileList if re.match(image_pattern, file)]
            if len(matched_files) < 1:
                continue
            for image_file in matched_files:
                caption_file = image_file[:-22] + str(caption_stim_index).zfill(5) + image_file[-17:]
                if caption_file in FileList:
                    beta_caption[count] = io.loadmat(os.path.join(subject_beta_root, caption_file))['beta']
                    beta_image[count] = io.loadmat(os.path.join(subject_beta_root, image_file))['beta']
                    count = count + 1
        beta_caption = beta_caption[0:count]
        beta_image = beta_image[0:count]

        R = np.zeros(num_vertices[subject])
        for vertice in tqdm(range(num_vertices[subject])):
            corr, _ = pearsonr(beta_caption[:, vertice], beta_image[:, vertice])
            R[vertice] = corr

        t_stat = np.zeros(num_vertices[subject])
        p_value = np.zeros(num_vertices[subject])
        for vertice in tqdm(range(num_vertices[subject])):
            t_stat[vertice], p_value[vertice] = stats.ttest_rel(beta_caption[:, vertice], beta_image[:, vertice])

        lh_data = {
            'Correlation': R[:hemi_vertices[i][0]],
            't_stat': t_stat[:hemi_vertices[i][0]],
            'p_value': p_value[:hemi_vertices[i][0]],
            'possibility': 1 - p_value[:hemi_vertices[i][0]],
            'positive_possibility': (1 - p_value[:hemi_vertices[i][0]]) * np.sign(t_stat[:hemi_vertices[i][0]]),
            'negative_possibility': -(1 - p_value[:hemi_vertices[i][0]]) * np.sign(t_stat[:hemi_vertices[i][0]])
        }
        rh_data = {
            'Correlation': R[hemi_vertices[i][0]:],
            't_stat': t_stat[hemi_vertices[i][0]:],
            'p_value': p_value[hemi_vertices[i][0]:],
            'possibility': 1 - p_value[hemi_vertices[i][0]:],
            'positive_possibility': (1 - p_value[hemi_vertices[i][0]:]) * np.sign(t_stat[hemi_vertices[i][0]:]),
            'negative_possibility': -(1 - p_value[hemi_vertices[i][0]:]) * np.sign(t_stat[hemi_vertices[i][0]:])
        }

        lh_df = pd.DataFrame(lh_data)
        rh_df = pd.DataFrame(rh_data)

        lh_df.to_csv(lh_savedir, index=False)
        rh_df.to_csv(rh_savedir, index=False)

        fig_dir = os.path.join(subject_result_root, 'cap_img_corr.png')
        plot_weight(R, subject, hemi_vertices[i], fig_dir, vmax=1)

        fig_dir = os.path.join(subject_result_root, 'cap_img_ttest.png')
        plot_weight(t_stat, subject, hemi_vertices[i], fig_dir, vmax=30)

    # Behavior t-test
    for sub_id, subject in enumerate(subjects):
        print('For %s' % subject)

        subject_beta_root = os.path.join(beta_root, subject, 'Stimulus', 'beta_zscore')
        subject_result_root = os.path.join(beta_root, subject, 'Stimulus', 'basic_analysis')

        if not os.path.exists(subject_result_root):
            os.makedirs(subject_result_root)

        if not os.path.exists(subject_beta_root) or len(os.listdir(subject_beta_root)) != 8800:
            continue

        match_data = np.zeros((8800, num_vertices[subject]))
        unmatch_data = np.zeros((8800, num_vertices[subject]))

        match_count = 0
        unmatch_count = 0
        FileList = os.listdir(subject_beta_root)
        for file in tqdm(FileList):
            if file[-5] != 'a':
                stim = int(file[-24:-19])
            else:
                stim = int(file[-22:-17])

            if Stimulus_index[stim][-3:] == 'jpg':
                continue

            dir = os.path.join(subject_beta_root, file)
            data = io.loadmat(dir)
            beta = np.squeeze(data['beta'])
            pair_unmatch = data['pair_unmatch'][0][0]

            if pair_unmatch == 1:
                unmatch_data[unmatch_count] = beta
                unmatch_count += 1
            else:
                match_data[match_count] = beta
                match_count += 1

        match_data = match_data[:match_count]
        unmatch_data = unmatch_data[:unmatch_count]

        t_stat = np.zeros(num_vertices[subject])
        p_value = np.zeros(num_vertices[subject])
        for vertice in tqdm(range(num_vertices[subject])):
            t_stat[vertice], p_value[vertice] = stats.ttest_ind(match_data[:, vertice], unmatch_data[:, vertice])

        lh_savedir = os.path.join(subject_result_root, subject + '_match_caption_lh.csv')
        rh_savedir = os.path.join(subject_result_root, subject + '_match_caption_rh.csv')

        lh_data = {
            't_stat': t_stat[:hemi_vertices[sub_id][0]],
            'p_value': p_value[:hemi_vertices[sub_id][0]],
            'possibility': 1 - p_value[:hemi_vertices[sub_id][0]],
            'positive_possibility': (1 - p_value[:hemi_vertices[sub_id][0]]) * np.sign(t_stat[:hemi_vertices[sub_id][0]]),
            'negative_possibility': -(1 - p_value[:hemi_vertices[sub_id][0]]) * np.sign(t_stat[:hemi_vertices[sub_id][0]])
        }
        rh_data = {
            't_stat': t_stat[hemi_vertices[sub_id][0]:],
            'p_value': p_value[hemi_vertices[sub_id][0]:],
            'possibility': 1 - p_value[hemi_vertices[sub_id][0]:],
            'positivie_possibility': (1 - p_value[hemi_vertices[sub_id][0]:]) * np.sign(t_stat[hemi_vertices[sub_id][0]:]),
            'negative_possibitiliy': -(1 - p_value[hemi_vertices[sub_id][0]:]) * np.sign(t_stat[hemi_vertices[sub_id][0]:])
        }

        lh_df = pd.DataFrame(lh_data)
        rh_df = pd.DataFrame(rh_data)

        lh_df.to_csv(lh_savedir, index=False)
        rh_df.to_csv(rh_savedir, index=False)

        fig_dir = os.path.join(subject_result_root, 'match_unmatch_cap_ttest.png')
        plot_weight(t_stat, subject, hemi_vertices[sub_id], fig_dir, vmax=10)

    for sub_id, subject in enumerate(subjects):
        print('For %s' % subject)

        subject_behavior_root = os.path.join(data_root, subject, 'behavior', 'stimulus')
        subject_script_root = os.path.join(script_root, 'subject' + str(sub_id + 1))
        subject_beta_root = os.path.join(beta_root, subject, 'Stimulus', 'beta_zscore')
        subject_result_root = os.path.join(beta_root, subject, 'Stimulus', 'basic_analysis')

        if not os.path.exists(subject_result_root):
            os.makedirs(subject_result_root)

        if not os.path.exists(subject_beta_root) or len(os.listdir(subject_beta_root)) != 8800:
            continue

        lh_savedir = os.path.join(subject_result_root, subject + '_match_image_lh.csv')
        rh_savedir = os.path.join(subject_result_root, subject + '_match_image_rh.csv')

        FileList = os.listdir(subject_behavior_root)

        match_data = np.zeros((8800, num_vertices[subject]))
        unmatch_data = np.zeros((8800, num_vertices[subject]))

        match_count = 0
        unmatch_count = 0
        for i in tqdm(range(1, 201)):
            for file in FileList:
                if 'run' + str(i) + '_' in file and '.mat' in file:
                    behavior_response = io.loadmat(os.path.join(subject_behavior_root, file))['theData'][0]['resp'][0][0]

            with open(os.path.join(subject_script_root, 'subject' + str(sub_id + 1) + '_run' + str(i) + '.txt'), 'r', encoding='gbk') as f:
                run_stim = f.readlines()
            f.close()

            for line_id, line in enumerate(run_stim):
                temp = line.split()
                if temp[0] == 'Index' or temp[4] == 'None':
                    continue
                stim_id = Stimulus_index[temp[4]]
                beta_dir = os.path.join(subject_beta_root,
                                        subject + '_stim_' + str(stim_id).zfill(5) + '_Run_' + str(i).zfill(3) + '_beta.mat')
                if behavior_response[line_id - 1] == 0:
                    match_data[match_count] = np.squeeze(io.loadmat(beta_dir)['beta'])
                    match_count += 1
                else:
                    unmatch_data[unmatch_count] = np.squeeze(io.loadmat(beta_dir)['beta'])
                    unmatch_count += 1

        match_data = match_data[:match_count]
        unmatch_data = unmatch_data[:unmatch_count]

        t_stat = np.zeros(num_vertices[subject])
        p_value = np.zeros(num_vertices[subject])
        for vertice in tqdm(range(num_vertices[subject])):
            t_stat[vertice], p_value[vertice] = stats.ttest_ind(match_data[:, vertice], unmatch_data[:, vertice])

        lh_data = {
            't_stat': t_stat[:hemi_vertices[sub_id][0]],
            'p_value': p_value[:hemi_vertices[sub_id][0]],
            'possibility': 1 - p_value[:hemi_vertices[sub_id][0]],
            'positive_possibility': (1 - p_value[:hemi_vertices[sub_id][0]]) * np.sign(t_stat[:hemi_vertices[sub_id][0]]),
            'negative_possibility': -(1 - p_value[:hemi_vertices[sub_id][0]]) * np.sign(t_stat[:hemi_vertices[sub_id][0]])
        }
        rh_data = {
            't_stat': t_stat[hemi_vertices[sub_id][0]:],
            'p_value': p_value[hemi_vertices[sub_id][0]:],
            'possibility': 1 - p_value[hemi_vertices[sub_id][0]:],
            'positive_possibility': (1 - p_value[hemi_vertices[sub_id][0]:]) * np.sign(t_stat[hemi_vertices[sub_id][0]:]),
            'negative_possibility': -(1 - p_value[hemi_vertices[sub_id][0]:]) * np.sign(t_stat[hemi_vertices[sub_id][0]:])
        }

        lh_df = pd.DataFrame(lh_data)
        rh_df = pd.DataFrame(rh_data)

        lh_df.to_csv(lh_savedir, index=False)
        rh_df.to_csv(rh_savedir, index=False)

        fig_dir = os.path.join(subject_result_root, 'match_unmatch_img_ttest.png')
        plot_weight(t_stat, subject, hemi_vertices[sub_id], fig_dir, vmax=10)
