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

    length = 18

    for sub_id, subject in enumerate(subjects):
        print(subject)
        sub_script_root = os.path.join(script_root, 'subject'+str(sub_id+1))
        sub_data_root = os.path.join(processed_root, subject, 'Stimulus', 'rescaled')
        savedir = os.path.join(processed_root, subject, 'Stimulus', 'basic_analysis', subject + '_ts_corr.csv')

        