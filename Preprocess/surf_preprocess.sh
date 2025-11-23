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

echo "Begin"
date
RootPath="/public/home/lishr2022/Project/Cross-modal/pipeline"

echo "工作目录:    $1"

ResultsPath=$1    #ResultsPath是数据处理后最终保存的位置
T1=$2
OutputF2S="${ResultsPath}/F2S"
OutputFL="${ResultsPath}/FL"
OutputSURF="${ResultsPath}/SURF"

AP_fl=${OutputFL}/AP_fMRIAfterfilterAddMean.nii.gz
PA_fl=${OutputFL}/PA_fMRIAfterfilterAddMean.nii.gz
RefAP=${OutputF2S}/AP_postvols/vol0_jac.nii.gz
RefPA=${OutputF2S}/PA_postvols/vol0_jac.nii.gz

echo "Recon all by FreeSurfer"
date
echo "########################################"
subject_name=$3
OutputREG="${ResultsPath}/REG"

# recon-all -i $T1 -subjid $subject_name -all -openmp 8

echo "Boundary-based register by FSL"
date
echo "########################################"
mkdir -p ${OutputREG}
bbregister --s ${subject_name} --mov $RefAP --init-fsl --reg ${OutputREG}/registerAP.dat --bold
bbregister --s ${subject_name} --mov $RefPA --init-fsl --reg ${OutputREG}/registerPA.dat --bold

echo "Resample the data onto the surface"
date
echo "########################################"

mkdir -p ${OutputSURF}

mri_vol2surf --mov ${AP_fl} --reg ${OutputREG}/registerAP.dat --projfrac 0.5 --interp nearest --hemi lh --o ${OutputSURF}/AP_lh_surf.mgh
mri_vol2surf --mov ${AP_fl} --reg ${OutputREG}/registerAP.dat --projfrac 0.5 --interp nearest --hemi rh --o ${OutputSURF}/AP_rh_surf.mgh

mri_vol2surf --mov ${PA_fl} --reg ${OutputREG}/registerPA.dat --projfrac 0.5 --interp nearest --hemi lh --o ${OutputSURF}/PA_lh_surf.mgh
mri_vol2surf --mov ${PA_fl} --reg ${OutputREG}/registerPA.dat --projfrac 0.5 --interp nearest --hemi rh --o ${OutputSURF}/PA_rh_surf.mgh
