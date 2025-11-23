import os
import shutil
import numpy as np
from scipy import io
import nibabel as nib
from tqdm import tqdm

if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    root = '/public/home/lishr2022/Project/Cross-modal/beta_estimate'
    Processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'
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

    for subject in subjects:
        print('For %s' % subject)
        beta_root = os.path.join(Processed_root, subject, 'Stimulus')
        group_text = os.path.join(root, subject+'.txt')
        temp = np.loadtxt(group_text)

        total_run, total_group = temp.shape
        session_index = np.zeros(total_run)
        count = 0
        for group in range(total_group):
            session_index[temp[:, group] > 0] = count + temp[temp[:, group] > 0, group]
            count = count + np.max(temp[:, group])
        session_index = session_index.astype(np.int64)

        sub_beta_root = os.path.join(beta_root, 'beta')
        sub_beta_target = os.path.join(beta_root, 'beta_zscore')

        if not os.path.exists(sub_beta_root):
            continue

        if not os.path.exists(sub_beta_target):
            os.makedirs(sub_beta_target)
        else:
            continue

        total = 0
        FileList = sorted(os.listdir(sub_beta_root))

        if len(FileList) != 8800:
            continue

        for i in tqdm(range(1, np.max(session_index)+1)):
            data = np.zeros((500, num_vertices[subject]))
            beta_list = []
            pair_stim = []
            count = 0
            for run in range(1, total_run+1):
                if session_index[run-1] != i:
                    continue
                for file in FileList:
                    if '_Run_' + str(run).zfill(3) + '_beta' in file:
                        beta_list.append(file)
                        data[count, :] = np.squeeze(io.loadmat(os.path.join(sub_beta_root, file))['beta_data'])

                        temp = np.squeeze(io.loadmat(os.path.join(sub_beta_root, file))['pair_stim'])
                        pair_stim.append(temp)

                        count = count + 1
            data = data[0:count, :]
            data_zscore = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

            for t, file in enumerate(beta_list):
                savedir = os.path.join(sub_beta_target, file)
                io.savemat(savedir, {'beta': data_zscore[t], 'pair_stim': pair_stim[t]})
