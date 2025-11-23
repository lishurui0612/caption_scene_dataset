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

subject_name=$1
T1=$2

recon-all -i $T1 -subjid $subject_name -all -openmp 8