module add apps/fsl/6.0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tats

export FREESURFER_HOME=/public/home/lishr2022/freesurfer
export SUBJECTS_DIR=/public/home/lishr2022/Project/Cross-modal/test_data/subjects
export FSFAST_HOME=/public/home/lishr2022/freesurfer/fsfast
export MNI_DIR=/public/home/lishr2022/freesurfer/mni
export FS_LICENSE=/public/home/lishr2022/freesurfer/license.txt

source $FSLDIR/etc/fslconf/fsl.sh
source $FREESURFER_HOME/SetUpFreeSurfer.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tats

echo "Begin"
date
RootPath="/public/home/lishr2022/Project/Cross-modal/Preprocess"

ResultsPath=$1
T1_1=$2
T1_2=$3
T1_3=$4
T1_4=$5

echo "Reoriented2std"
date
echo "######################################"
fslreorient2std $T1_1 $T1_1
fslreorient2std $T1_2 $T1_2
fslreorient2std $T1_3 $T1_3
fslreorient2std $T1_4 $T1_4

echo "Brain Extraction"
date
echo "######################################"
echo "T1_1"
bet ${T1_1} ${ResultsPath}/T1_bet1 -R -B -m -f 0.3
#fast -t 1 -n 3 -g ${ResultsPath}/T1_bet1

echo "T1_2"
bet ${T1_2} ${ResultsPath}/T1_bet2 -R -B -m -f 0.3
#fast -t 1 -n 3 -g ${ResultsPath}/T1_bet2

echo "T1_3"
bet ${T1_3} ${ResultsPath}/T1_bet3 -R -B -m -f 0.3
#fast -t 1 -n 3 -g ${ResultsPath}/T1_bet3

echo "T1_4"
bet ${T1_4} ${ResultsPath}/T1_bet4 -R -B -m -f 0.3
#fast -t 1 -n 3 -g ${ResultsPath}/T1_bet4

echo "Flirt between T1_1 & T1_2"
date
echo "######################################"
flirt -ref ${ResultsPath}/T1_bet1.nii.gz -in ${ResultsPath}/T1_bet2.nii.gz -out ${ResultsPath}/T1_bet2_f.nii.gz -dof 6 -omat ${ResultsPath}/T1_2to1.mat
applywarp -r ${T1_1} -i ${T1_2}  -o ${ResultsPath}/T1_RUN_2_f.nii.gz --premat=${ResultsPath}/T1_2to1.mat --interp=spline

echo "Flirt between T1_1 & T1_3"
date
echo "######################################"
flirt -ref ${ResultsPath}/T1_bet1.nii.gz -in ${ResultsPath}/T1_bet3.nii.gz -out ${ResultsPath}/T1_bet3_f.nii.gz -dof 6 -omat ${ResultsPath}/T1_3to1.mat
applywarp -r ${T1_1} -i ${T1_3}  -o ${ResultsPath}/T1_RUN_3_f.nii.gz --premat=${ResultsPath}/T1_3to1.mat --interp=spline

echo "Flirt between T1_1 & T1_4"
date
echo "######################################"
flirt -ref ${ResultsPath}/T1_bet1.nii.gz -in ${ResultsPath}/T1_bet4.nii.gz -out ${ResultsPath}/T1_bet4_f.nii.gz -dof 6 -omat ${ResultsPath}/T1_4to1.mat
applywarp -r ${T1_1} -i ${T1_4}  -o ${ResultsPath}/T1_RUN_4_f.nii.gz --premat=${ResultsPath}/T1_4to1.mat --interp=spline

echo "Compute Average T1 image"
date
echo "######################################"
#fslmaths ${T1_1} -add ${ResultsPath}/T1_RUN_2_f.nii.gz -add ${ResultsPath}/T1_RUN_3_f.nii.gz -add ${ResultsPath}/T1_RUN_4_f.nii.gz -div 4 ${ResultsPath}/T1_avg.nii.gz
fslmaths ${T1_1} -add ${ResultsPath}/T1_RUN_2_f.nii.gz -add ${ResultsPath}/T1_RUN_4_f.nii.gz -div 3 ${ResultsPath}/T1_avg.nii.gz