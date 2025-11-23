# coding=gbk
import os
import cv2
import imageio
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Data'
    processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'

    for subject in subjects:
        print(subject)
        savedir = os.path.join(processed_root, subject, 'quality_control', 'register')
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        func_dir = os.path.join(root, subject, 'func', 'stimulus')
        FileList = sorted(os.listdir(func_dir))

        count = 0
        for file in tqdm(FileList):
            count += 1
            AP_dir = os.path.join(func_dir, file, 'AP_fMRIAfterMinP.nii.gz')
            if os.path.exists(AP_dir):
                fig = plt.figure(figsize=(9, 3))
                fmri_img = nib.load(AP_dir).get_fdata()
                Ref_img = fmri_img[:, :, :, 0]

                fmri_slice = Ref_img[:, :, Ref_img.shape[2]//2+10]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 1)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                fmri_slice = Ref_img[:, Ref_img.shape[1]//2, :]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 2)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                fmri_slice = Ref_img[Ref_img.shape[0]//2, :, :]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 3)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                plt.title('Run '+ str(count).zfill(3))
                fig.savefig(os.path.join(savedir, subject+'_'+str(count).zfill(3)+'.png'), bbox_inches='tight', dpi=300)
                plt.close(fig)

            count += 1
            PA_dir = os.path.join(func_dir, file, 'PA_fMRIAfterMinP.nii.gz')
            if os.path.exists(PA_dir):
                fig = plt.figure(figsize=(9, 3))
                fmri_img = nib.load(PA_dir).get_fdata()
                Ref_img = fmri_img[:, :, :, 0]

                fmri_slice = Ref_img[:, :, Ref_img.shape[2] // 2 + 10]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 1)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                fmri_slice = Ref_img[:, Ref_img.shape[1] // 2, :]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 2)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                fmri_slice = Ref_img[Ref_img.shape[0] // 2, :, :]
                fmri_slice = np.rot90(fmri_slice)
                ax = fig.add_subplot(1, 3, 3)
                ax.imshow(fmri_slice, cmap="gray", vmin=-60, vmax=1500)  # 可以调整透明度
                ax.axis("off")

                plt.title('Run ' + str(count).zfill(3))
                fig.savefig(os.path.join(savedir, subject + '_' + str(count).zfill(3) + '.png'), bbox_inches='tight', dpi=300)
                plt.close(fig)

        frames = []
        FileList = sorted(os.listdir(savedir))
        while len(FileList) < 200:
            FileList = FileList + FileList
        FileList = FileList[:200]
        for file in tqdm(FileList):
            frames.append(cv2.imread(os.path.join(savedir, file)))
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(
            os.path.join(savedir, subject+'_mc.mp4'),
            cv2.VideoWriter_fourcc(*"mp4v"),  # 编码格式
            10,  # 帧率，每秒 2 帧
            (width, height),
        )

        for frame in tqdm(frames):
            video_writer.write(frame)

        video_writer.release()
