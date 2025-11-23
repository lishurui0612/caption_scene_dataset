echo "Begin"
date
RootPath="/public/home/lishr2022/Project/Cross-modal/Preprocess"
DataPath="/public_bme2/bme-liyuanning/lishr/Cross_modal/Data"

subject=$1
T1="${DataPath}/${subject}/anat/T1/T1_avg.nii.gz"

echo "register Str to MNI"
date
echo "########################################"

OutputSEG="${DataPath}/${subject}/anat/T1/SEG"
mkdir -p ${OutputSEG}

bet ${T1} ${OutputSEG}/T1_bet -R -B -m -f 0.3

T1_bet="${OutputSEG}/T1_bet.nii.gz"

RefImgforOneStep=${OutputSEG}/T1DownResample.nii.gz
flirt -in ${T1_bet} -ref ${T1_bet} -o ${RefImgforOneStep} -applyisoxfm 2.5 -interp nearestneighbour
fslmaths $RefImgforOneStep -bin "${OutputSEG}/BrainMask"

OutputS2M="${DataPath}/${subject}/anat/T1/S2M"
mkdir -p ${OutputS2M}

MNILabelImage="$RootPath/template_and_acqparams/label_template_mni_sym_addsymsubcort_masked_2mm.nii"
antsRegistrationSyNQuick.sh -d 3 -f ${MNILabelImage} -m ${RefImgforOneStep} -o ${OutputS2M}/Str2MNIAnt
antsApplyTransforms -d 3 -i ${RefImgforOneStep} -r ${MNILabelImage} -o ${OutputS2M}/Str2MNIAntWarped_NN.nii.gz -t ${OutputS2M}/Str2MNIAnt1Warp.nii.gz -t ${OutputS2M}/Str2MNIAnt0GenericAffine.mat  --interpolation NearestNeighbor
ComposeMultiTransform 3  ${OutputS2M}/Str2MNI.nii.gz -R ${MNILabelImage} ${OutputS2M}/Str2MNIAnt1Warp.nii.gz ${OutputS2M}/Str2MNIAnt0GenericAffine.mat
antsApplyTransforms -d 3 -i ${RefImgforOneStep} -r ${MNILabelImage} -o ${OutputS2M}/Str2MNIAntWarped_NN2.nii.gz -t ${OutputS2M}/Str2MNI.nii.gz  --interpolation NearestNeighbor

#input='/public_bme2/bme-liyuanning/lishr/caption_secnes_dataset/derivative/pp_data/sub-01/func/sub-01_ses-01_task-fLoc_run-001.nii.gz'
#output='/public_bme2/bme-liyuanning/lishr/caption_secnes_dataset/sub-01_ses-01_task-fLoc_run-001.nii.gz'
#antsApplyTransforms -e 3 -i ${input} -r ${MNILabelImage} -o ${output} -t ${OutputS2M}/Str2MNI.nii.gz --interpolation BSpline --float
