import os
import argparse
import subprocess
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)

    args = parser.parse_args()

    file_list = os.listdir(os.path.join(args.path, 'SURF'))
    if len(file_list) != 4:
        OutputSURF = os.path.join(args.path, 'SURF')
        if not os.path.exists(OutputSURF):
            os.makedirs(OutputSURF)

        AP_minP = os.path.join(args.path, 'AP_fMRIAfterMinP.nii.gz')

        AP_lh = os.path.join(OutputSURF, 'AP_lh_surf.mgh')
        AP_rh = os.path.join(OutputSURF, 'AP_rh_surf.mgh')
        subprocess.run([
            'mri_vol2surf',  '--src', AP_minP, '--out', AP_lh, '--regheader', args.subject,
            '--projfrac-avg', '0', '1', '0.2', '--interp', 'trilinear', '--hemi', 'lh'
        ])

        subprocess.run([
            'mri_vol2surf',  '--src', AP_minP, '--out', AP_rh, '--regheader', args.subject,
            '--projfrac-avg', '0', '1', '0.2', '--interp', 'trilinear', '--hemi', 'rh'
        ])

        PA_minP = os.path.join(args.path, 'PA_fMRIAfterMinP.nii.gz')

        PA_lh = os.path.join(OutputSURF, 'PA_lh_surf.mgh')
        PA_rh = os.path.join(OutputSURF, 'PA_rh_surf.mgh')
        subprocess.run([
            'mri_vol2surf', '--src', PA_minP, '--out', PA_lh, '--regheader', args.subject,
            '--projfrac-avg', '0', '1', '0.2', '--interp', 'trilinear', '--hemi', 'lh'
        ])

        subprocess.run([
            'mri_vol2surf', '--src', PA_minP, '--out', PA_rh, '--regheader', args.subject,
            '--projfrac-avg', '0', '1', '0.2', '--interp', 'trilinear', '--hemi', 'rh'
        ])