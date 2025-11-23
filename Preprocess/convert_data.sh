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

inputdir=$1
outputdir=$2

if [ ! -d ${outputdir} ]; then
  mkdir -p ${outputdir}
fi

dcm2niix -o $2 -z y $1