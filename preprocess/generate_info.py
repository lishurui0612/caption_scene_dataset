import os
import imageio
import argparse
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from skimage import filters

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--rootpath", type=str, required=True)
parser.add_argument("--dura", type=float, default=1.5)
parser.add_argument("--dim", type=int, default=4)

args = parser.parse_args()


def scale_data(data):
    p10 = np.percentile(data, 10)
    data[data<p10] = p10
    data -= p10

    p99 = np.percentile(data, 99.5)
    data[data>p99] = p99
    data /= p99
    data *= 255

    return data


def show_AP_PA(AP_path, PA_path, save_path):
    data_ap = nib.load(AP_path)
    data_ap = nib.as_closest_canonical(data_ap)
    data_ap = data_ap.get_fdata()
    # data_ap-=np.percentile(data_ap, 10)
    # data_ap/=np.percentile(data_ap, 99.9)
    # data_ap*=255
    # data_ap[data_ap>255]=255
    # data_ap[data_ap<0]=0
    data_ap = scale_data(data_ap)

    data_pa = nib.load(PA_path)
    data_pa = nib.as_closest_canonical(data_pa)
    data_pa = data_pa.get_fdata()
    # data_pa-=np.percentile(data_pa, 10)
    # data_pa/=np.percentile(data_pa, 99.9)
    # data_pa*=255
    # data_pa[data_pa>255]=255
    # data_pa[data_pa<0]=0
    data_pa = scale_data(data_pa)

    # data_diff = data_ap - data_pa

    # data_diff /= np.percentile(data_diff, 99.9)
    # data_diff[data_diff>1] = 1
    # data_diff[data_diff<-1] = -1
    # data_diff*=255

    dim_1, dim_2, dim_3 = data_ap.shape
    col = int(np.ceil(dim_3/8))
    data_big = np.zeros([dim_2*8, col*dim_1, 3])
    for idx in range(dim_3):
        data_big[(idx//col)*dim_2:(idx//col+1)*dim_2, (idx%col)*dim_1:(idx%col+1)*dim_1, 0] = data_ap[:, ::-1, idx].T
        data_big[(idx//col)*dim_2:(idx//col+1)*dim_2, (idx%col)*dim_1:(idx%col+1)*dim_1, 1] = np.minimum(data_ap[:, ::-1, idx].T, data_pa[:, ::-1, idx].T)
        data_big[(idx//col)*dim_2:(idx//col+1)*dim_2, (idx%col)*dim_1:(idx%col+1)*dim_1, 2] = data_pa[:, ::-1, idx].T

    data_big = data_big.astype(np.uint8)
    cv2.imwrite(save_path, data_big)


def get_mc_info(path, fig_path):
    mc_data = np.loadtxt(path)
    jd_max = np.max(np.abs(mc_data[:, 0:3])*180/np.pi)
    wy_max = np.max(np.abs(mc_data[:, 3:6]))

    fig, axes = plt.subplots()
    axes.set_ylim((-3, 3))

    color = 'tab:blue'
    axes.set_xlabel('Time point')
    axes.set_ylabel('rotate', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 0]*180/np.pi, label='rotate 1', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 1]*180/np.pi, label='rotate 2', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 2]*180/np.pi, label='rotate 3', color=color)
    axes.legend(loc='upper left')
    axes.tick_params(axis='y', labelcolor=color)

    axes = axes.twinx()
    axes.set_ylim((-3, 3))

    color = 'tab:green'
    axes.set_ylabel('moving', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 3], label='moving 1', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 4], label='moving 2', color=color)
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 5], label='moving 3', color=color)
    axes.legend(loc='upper right')
    axes.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(fig_path)

    return jd_max, wy_max


def plot_mc(path_par, path_fd, fig_path):
    mc_data = np.loadtxt(path_par)
    fd_data = np.loadtxt(path_fd)

    mc_data -= mc_data[0, :]

    jd_max = np.max(np.abs(mc_data[:, 0:3])*180/np.pi)
    wy_max = np.max(np.abs(mc_data[:, 3:6]))

    fig, axes = plt.subplots()
    axes.set_ylim((-4, 4))

    axes.set_xlabel('Time point')
    axes.set_ylabel('rotation', color='blue')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 0]*180/np.pi, label='rotation x', alpha = 0.5, color='blue')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 1]*180/np.pi, label='rotation y', alpha = 0.5, color='lightsteelblue')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 2]*180/np.pi, label='rotation z', alpha = 0.5, color='mediumslateblue')
    axes.legend(loc='upper left')
    axes.tick_params(axis='y', labelcolor='blue')

    axes = axes.twinx()
    axes.set_ylim((-4, 4))

    axes.set_ylabel('translation', color='green')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 3], label='translation x', alpha = 0.5, color='green')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 4], label='translation y', alpha = 0.5, color='gold')
    axes.plot(range(1, 1+mc_data.shape[0]), mc_data[:, 5], label='translation z', alpha = 0.5, color='darkorange')
    axes.plot(range(1, 1+mc_data.shape[0]), fd_data, label='framewise displacement', alpha = 0.5, color='red')
    axes.plot([1, mc_data.shape[0]], [3, 3], color='tomato', alpha = 0.5, linestyle="--")
    axes.plot([1, mc_data.shape[0]], [-3, -3], color='tomato', alpha = 0.5, linestyle="--")
    axes.plot([1, mc_data.shape[0]], [0.5, 0.5], color='lightsalmon', alpha = 0.5, linestyle="--")
    axes.plot([1, mc_data.shape[0]], [-0.5, -0.5], color='lightsalmon', alpha = 0.5, linestyle="--")
    axes.legend(loc='upper right')
    axes.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return jd_max, wy_max


def main(args):
    show_AP_PA(args.path + "/SBRefAP.nii.gz", args.path + "/SBRefPA.nii.gz",
               args.path + '/FIG/fig1_before_topup.png')
    show_AP_PA(args.path + "/DC/SBRefAP_dc_jac.nii.gz", args.path + "/DC/SBRefPA_dc_jac.nii.gz",
               args.path + '/FIG/fig2_after_topup.png')

    if os.path.exists(args.path + "/AP_st.nii.gz"):
        AP_jd, AP_wy = plot_mc(args.path + "/MC/AP_st_dc_mc.par", args.path + "/MC/PhaseAP_FD.txt",
                               args.path + '/FIG/fig3_MC_AP.png')
    if os.path.exists(args.path + "/PA_st.nii.gz"):
        PA_jd, PA_wy = plot_mc(args.path + "/MC/PA_st_dc_mc.par", args.path + "/MC/PhasePA_FD.txt",
                               args.path + '/FIG/fig4_MC_PA.png')


main(args)
