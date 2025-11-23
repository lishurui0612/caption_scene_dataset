echo "Begin"
date
RootPath="/public/home/lishr2022/Project/Cross-modal/Preprocess"

ResultsPath=$1
T2_1=$2
T2_2=$3
T1_root=$4

echo "Reoriented2std"
date
echo "######################################"
fslreorient2std $T2_1 $T2_1
fslreorient2std $T2_2 $T2_2

echo "Brain Extraction"
date
echo "######################################"
echo "T1_avg"
bet ${T1_root}/T1_avg.nii.gz ${T1_root}/T1_avg_bet -B -m -f 0.5
fast -t 1 -n 3 -g ${T1_root}/T1_avg_bet

echo "T2_1"
bet ${T2_1} ${ResultsPath}/T2_bet1 -B -m -f 0.5

echo "T2_2"
bet ${T2_2} ${ResultsPath}/T2_bet2 -B -m -f 0.5

echo "Flirt between T2_1 & T2_2"
date
echo "######################################"
flirt -ref ${ResultsPath}/T2_bet1.nii.gz -in ${ResultsPath}/T2_bet2.nii.gz -out ${ResultsPath}/T2_bet2_f.nii.gz -dof 6 -omat ${ResultsPath}/T2_2to1.mat
applywarp -r ${T2_1} -i ${T2_2}  -o ${ResultsPath}/T2_RUN_2_f.nii.gz --premat=${ResultsPath}/T2_2to1.mat --interp=spline

echo "Compute Average T2 image"
date
echo "######################################"
fslmaths ${T2_1} -add ${ResultsPath}/T2_RUN_2_f.nii.gz -div 2 ${ResultsPath}/T2_avg.nii.gz

bet ${ResultsPath}/T2_avg.nii.gz ${ResultsPath}/T2_avg_bet -B -m -f 0.5

echo "Flirt between T1_avg & T1_avg"
date
echo "######################################"
flirt -ref ${T1_root}/T1_avg_bet.nii.gz -in ${ResultsPath}/T2_avg_bet.nii.gz -out ${ResultsPath}/T2_bet2T1.nii.gz -dof 6 -omat ${ResultsPath}/T2_2T1.mat
applywarp -r ${T1_root}/T1_avg.nii.gz -i ${ResultsPath}/T2_avg.nii.gz -o ${ResultsPath}/T2_2T1.nii.gz --premat=${ResultsPath}/T2_2T1.mat --interp=spline

echo "Brain Extraction"
date
echo "######################################"
bet ${ResultsPath}/T2_2T1.nii.gz ${ResultsPath}/T2_2T1bet -B -m -f 0.5