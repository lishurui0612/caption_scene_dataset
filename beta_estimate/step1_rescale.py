import os
import shutil
import numpy as np
from scipy import io
import nibabel as nib
from tqdm import tqdm

subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
root = '/public_bme2/bme-liyuanning/lishr/Cross_modal'
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

for subject in subjects:
    subject_root = os.path.join(root, 'Data', subject, 'func/stimulus')
    subject_target = os.path.join(root, 'Processed_Data', subject, 'Stimulus/min_process')
    print('For %s' % subject)
    # 检查是否进行rescale
    flag = 1
    zscore_dir = os.path.join(root, 'Processed_Data', subject, 'Stimulus', 'rescaled')
    if os.path.exists(zscore_dir):
        flag = 0
    for run in range(1, 200, 2):
        surf_dir = os.path.join(subject_root, 'Run_'+str(run).zfill(3)+'_'+str(run+1).zfill(3), 'SURF')
        if not os.path.exists(surf_dir):
            flag = 0
            print('For run %d, there is something wrong!' % run)
            break
        file_list = os.listdir(surf_dir)
        if len(file_list) != 4:
            flag = 0
            print('For run %d, there is something wrong!' % run)
            break
    if flag == 0:
        continue


    if not os.path.exists(subject_target):
        os.makedirs(subject_target)

    count = 0
    for run in tqdm(range(1, 200, 2)):
        run_dir = os.path.join(subject_root, 'Run_' + str(run).zfill(3) + '_' + str(run+1).zfill(3))

        if (subject == 'S1_LiJiawei' and run >= 43 and run <= 47) or (subject == 'S5_FangYu' and run >= 13 and run <= 18):
            # 处理PA
            count = count + 1
            lh_sourcedir = os.path.join(run_dir, 'SURF/PA_lh_surf.mgh')
            lh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_lh.mgh')
            shutil.copy(lh_sourcedir, lh_targetdir)

            rh_sourcedir = os.path.join(run_dir, 'SURF/PA_rh_surf.mgh')
            rh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_rh.mgh')
            shutil.copy(rh_sourcedir, rh_targetdir)

            # 处理AP
            count = count + 1
            lh_sourcedir = os.path.join(run_dir, 'SURF/AP_lh_surf.mgh')
            lh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_lh.mgh')
            shutil.copy(lh_sourcedir, lh_targetdir)

            rh_sourcedir = os.path.join(run_dir, 'SURF/AP_rh_surf.mgh')
            rh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_rh.mgh')
            shutil.copy(rh_sourcedir, rh_targetdir)
        else:
            # 处理AP
            count = count + 1
            lh_sourcedir = os.path.join(run_dir, 'SURF/AP_lh_surf.mgh')
            lh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_lh.mgh')
            shutil.copy(lh_sourcedir, lh_targetdir)

            rh_sourcedir = os.path.join(run_dir, 'SURF/AP_rh_surf.mgh')
            rh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_rh.mgh')
            shutil.copy(rh_sourcedir, rh_targetdir)

            # 处理PA
            count = count + 1
            lh_sourcedir = os.path.join(run_dir, 'SURF/PA_lh_surf.mgh')
            lh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_lh.mgh')
            shutil.copy(lh_sourcedir, lh_targetdir)

            rh_sourcedir = os.path.join(run_dir, 'SURF/PA_rh_surf.mgh')
            rh_targetdir = os.path.join(subject_target, subject + '_Run_' + str(count).zfill(3) + '_rh.mgh')
            shutil.copy(rh_sourcedir, rh_targetdir)


for subject in subjects:
    print('For %s' % subject)
    subject_root = os.path.join(root, 'Processed_Data', subject, 'Stimulus/min_process')
    subject_target = os.path.join(root, 'Processed_Data', subject, 'Stimulus/mat')

    if not os.path.exists(subject_root):
        continue

    if not os.path.exists(subject_target):
        os.makedirs(subject_target)
    else:
        continue

    for run in tqdm(range(1, 201)):
        lh_root = os.path.join(subject_root, subject + '_Run_' + str(run).zfill(3) + '_lh.mgh')
        rh_root = os.path.join(subject_root, subject + '_Run_' + str(run).zfill(3) + '_rh.mgh')

        lh_data = np.squeeze(nib.load(lh_root).get_fdata())
        rh_data = np.squeeze(nib.load(rh_root).get_fdata())

        whole_data = np.concatenate((lh_data, rh_data), axis=0)

        mat_target = os.path.join(subject_target, subject + '_Run_' + str(run).zfill(3) + '.mat')

        io.savemat(mat_target, {'data': whole_data})

    shutil.rmtree(subject_root)


for index, subject in enumerate(subjects):
    print('For %s' % subject)
    # 读取分组文件
    group_txt = os.path.join('/public/home/lishr2022/Project/Cross-modal/beta_estimate', subject+'.txt')
    subject_root = os.path.join(root, 'Processed_Data', subject, 'Stimulus/mat')
    rescaled_root = os.path.join(root, 'Processed_Data', subject, 'Stimulus/rescaled')

    if not os.path.exists(subject_root):
        continue

    if not os.path.exists(rescaled_root):
        os.makedirs(rescaled_root)
    else:
        continue

    with open(group_txt, 'r') as f:
        group = np.loadtxt(group_txt).astype('uint')
    f.close()

    if np.sum(group > 0) != 200:
        print('Group has something wrong!')
        break

    Parameter = []
    Group_Parameter = []
    total_run, total_group = group.shape
    for group_id in range(total_group):
        temp_group = np.squeeze(group[:, group_id])
        # 计算平均帧
        print('Now is calculating the mean volume for group %d' % (group_id+1))
        mean = np.zeros((num_vertices[subject], 1))
        for run in range(total_run):
            if temp_group[run] > 0:
                run_data = io.loadmat(os.path.join(subject_root, subject + '_Run_' + str(run+1).zfill(3) + '.mat'))['data']
                mean = mean + run_data
        mean = mean / np.sum(temp_group > 0)

        # Linear Regression to rescale the data within session
        Group_Parameter = []
        for session in range(np.max(temp_group)):
            print('Linear regression for group %d, session %d' % (group_id+1, session+1))
            indices = np.where(temp_group == session+1)[0] + 1

            # concatenate the data
            run_data = io.loadmat(os.path.join(subject_root, subject + '_Run_' + str(indices[0]).zfill(3) + '.mat'))['data']
            run_data = run_data[np.newaxis, :]
            group_data = run_data
            for i in indices[1:]:
                run_data = io.loadmat(os.path.join(subject_root, subject + '_Run_' + str(i).zfill(3) + '.mat'))['data']
                run_data = run_data[np.newaxis, :]
                group_data = np.concatenate((group_data, run_data), axis=0)
            print("The shape of the group_data is ", group_data.shape)

            # Close form solution for linear regression
            c, v, t = group_data.shape
            n = c * v * t
            xy = np.sum(group_data * mean)
            x_squared = np.sum(group_data * group_data)
            mean_x = np.mean(group_data)
            mean_y = np.mean(mean)
            a = (xy - n * mean_x * mean_y) / (x_squared - n * mean_x * mean_x)
            b = mean_y - a * mean_x

            Group_Parameter.append([a, b])
            print('Complete computing linear regression for Group %d Session %d. a = %.3f, b = %.3f'
                  % (group_id+1, session+1, a, b))

            # Save the rescaled data
            for i, run in enumerate(indices):
                rescaled_target = os.path.join(rescaled_root, subject + '_Run_' + str(run).zfill(3) + '_scaled.mat')
                temp = np.squeeze(group_data[i])
                temp = a * temp + b

                io.savemat(rescaled_target, {'data': temp})
                print('Successfully save Run %d' % (run))
    Parameter.append(Group_Parameter)
    shutil.rmtree(subject_root)
