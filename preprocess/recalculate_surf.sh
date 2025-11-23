module add apps/fsl/6.0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tats

export FREESURFER_HOME=/public/home/lishr2022/freesurfer
export SUBJECTS_DIR=/public_bme/data/lishr/Cross_modal/subjects
export FSFAST_HOME=/public/home/lishr2022/freesurfer/fsfast
export MNI_DIR=/public/home/lishr2022/freesurfer/mni
export FS_LICENSE=/public/home/lishr2022/freesurfer/license.txt

source $FSLDIR/etc/fslconf/fsl.sh
source $FREESURFER_HOME/SetUpFreeSurfer.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tats

ResultsPath=$1
subject_name=$2
OutputSURF="${ResultsPath}/SURF"
AP_minP=${ResultsPath}/AP_fMRIAfterMinP.nii.gz
PA_minP=${ResultsPath}/PA_fMRIAfterMinP.nii.gz

if [ -d $OutputSURF ]; then
  rm -rf $OutputSURF
fi

echo $ResultsPath
echo 'Recalculate the SURF'

mkdir -p $OutputSURF

mri_vol2surf --src ${AP_minP} --out ${OutputSURF}/AP_lh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi lh
mri_vol2surf --src ${AP_minP} --out ${OutputSURF}/AP_rh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi rh
mri_vol2surf --src ${PA_minP} --out ${OutputSURF}/PA_lh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi lh
mri_vol2surf --src ${PA_minP} --out ${OutputSURF}/PA_rh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi rh
