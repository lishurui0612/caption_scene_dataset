# coding=gbk
import os
import imageio
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    root = '/public_bme/data/lishr/Cross_modal/Data'
    processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'

    vertices = [149079 + 151166,
                135103 + 135723,
                155295 + 151303,
                141922 + 142796,
                141578 + 138836,
                146440 + 149139,
                145747 + 144531,
                129958 + 128115]

    for sub_id, subject in enumerate(subjects):
        print(subject)
        savedir = os.path.join(processed_root, subject, 'quality_control', subject+'_mean_surf.npy')

        mean = np.zeros((200, vertices[sub_id]))

        func_dir = os.path.join(processed_root, subject, 'Stimulus', 'rescaled')
        FileList = sorted(os.listdir(func_dir))

        count = -1
        for file in tqdm(FileList):
            count += 1

            data = np.squeeze(io.loadmat(os.path.join(func_dir, file))['data'])
            mean[count] = np.mean(data, axis=1)

        np.save(savedir, mean)
