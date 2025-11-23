echo "Begin"
date
RootPath="/public/home/lishr2022/Project/Cross-modal/Preprocess"

# 修改
echo "工作目录:    $1"

# 修改
ResultsPath=$1  #ResultsPath是数据处理后最终保存的位置
T1=$2
PhaseAP=$3
PhasePA=$4
starttime=$5
num_volumes=$6
subject=$7

subject_name=${subject:3}

cutAP=${ResultsPath}/cutAP.nii.gz
cutPA=${ResultsPath}/cutPA.nii.gz
SBRefAP=${ResultsPath}/SBRefAP.nii.gz
SBRefPA=${ResultsPath}/SBRefPA.nii.gz

echo reorient2std
echo "########################################"
fslreorient2std $T1 $T1
fslreorient2std $PhaseAP $PhaseAP
fslreorient2std $PhasePA $PhasePA

##########保存中间结果的子文件夹##########

OutputDC="${ResultsPath}/DC"
OutputMC="${ResultsPath}/MC"
OutputSEG="${ResultsPath}/SEG"
OutputF2S="${ResultsPath}/F2S"
OutputFL="${ResultsPath}/FL"
OutputFIG="${ResultsPath}/FIG"
OutputREG="${ResultsPath}/REG"
OutputSURF="${ResultsPath}/SURF"

##########提取中间帧#########

echo "Trim fMRI responses"

fslroi ${PhaseAP} ${cutAP} $starttime $num_volumes
fslroi ${PhasePA} ${cutPA} $starttime $num_volumes

##########Slice Timing Correction#########

echo "Slice timing correction"
python ${RootPath}/extract_ST.py --path ${ResultsPath}
slicetimer -i ${cutAP} -o ${ResultsPath}/AP_st -r 2 --tcustom=${ResultsPath}/ST_AP.txt
slicetimer -i ${cutPA} -o ${ResultsPath}/PA_st -r 2 --tcustom=${ResultsPath}/ST_PA.txt

AP_st=${ResultsPath}/AP_st.nii.gz
PA_st=${ResultsPath}/PA_st.nii.gz

##########提取平均帧作为参考########

echo "Select SBRef images"

fslmaths ${AP_st} -Tmean ${SBRefAP}
fslmaths ${PA_st} -Tmean ${SBRefPA}

##########topup矫正##########

echo "EPI distortion Correction using topup"
date
echo "########################################"

mkdir -p ${OutputDC}

fslmerge -t ${OutputDC}/BothPhases ${SBRefAP} ${SBRefPA}
fslmaths ${SBRefAP} -mul 0 -add 1 ${OutputDC}/Mask

# Read Topup configuration files
TopupConfig="$RootPath/template_and_acqparams/b02b0.cnf"
txtfname="$RootPath/template_and_acqparams/acqparams_cross_modal.txt"
numslice=`fslval ${OutputDC}/BothPhases dim3`

echo "epi distortion correction"
# Use topup function to do distortion correction
fslmaths ${OutputDC}/BothPhases -abs -add 1 -mas ${OutputDC}/Mask -dilM -dilM -dilM -dilM -dilM ${OutputDC}/BothPhases
topup --imain=${OutputDC}/BothPhases --datain=$txtfname --config=$TopupConfig --out=${OutputDC}/Coefficents --iout=${OutputDC}/Magnitudes --fout=${OutputDC}/TopupField --dfout=${OutputDC}/WarpField --rbmout=${OutputDC}/MotionMatrix --jacout=${OutputDC}/Jacobian -v

echo "AP"
convertwarp --relout --rel -r $SBRefAP --premat=${OutputDC}/MotionMatrix_01.mat --warp1=${OutputDC}/WarpField_01 --out=${OutputDC}/SBRefAP2FMWarpField.nii.gz
applywarp --rel --interp=spline -i ${SBRefAP} -r ${SBRefAP} -w ${OutputDC}/SBRefAP2FMWarpField.nii.gz -o ${OutputDC}/SBRefAP_dc.nii.gz
fslmaths ${OutputDC}/SBRefAP_dc.nii.gz -mul ${OutputDC}/Jacobian_01.nii.gz ${OutputDC}/SBRefAP_dc_jac.nii.gz

echo "PA"
convertwarp --relout --rel -r $SBRefPA --premat=${OutputDC}/MotionMatrix_02.mat --warp1=${OutputDC}/WarpField_02 --out=${OutputDC}/SBRefPA2FMWarpField.nii.gz
applywarp --rel --interp=spline -i ${SBRefPA} -r ${SBRefPA} -w ${OutputDC}/SBRefPA2FMWarpField.nii.gz -o ${OutputDC}/SBRefPA_dc.nii.gz
fslmaths ${OutputDC}/SBRefPA_dc.nii.gz -mul ${OutputDC}/Jacobian_02.nii.gz ${OutputDC}/SBRefPA_dc_jac.nii.gz

##########拆分4D fMRI为3D fMRI##########

echo "Split 4D fMRI to 3D fMRI"

if [ -f "$AP_st" ]; then
  mkdir -p ${OutputDC}/AP_prevols
  fslsplit ${AP_st} ${OutputDC}/AP_prevols/vol -t
fi

if [ -f "$PA_st" ]; then
  mkdir -p ${OutputDC}/PA_prevols
  fslsplit ${PA_st} ${OutputDC}/PA_prevols/vol -t
fi

##########Distortion correction for whole volumes##########

echo "Distortion correction for whole volumes"

if [ -f "$AP_st" ]; then
  echo "AP--------->"

  mkdir -p ${OutputDC}/AP_postvols

  TR_vol_AP=`${FSLDIR}/bin/fslval ${AP_st} pixdim4 | cut -d " " -f 1`
  NumFrames_AP=`${FSLDIR}/bin/fslval ${AP_st} dim4`

  FrameMergeDC_AP=""
  k=0
  while [ $k -lt $NumFrames_AP ]; do
    vnum=`${FSLDIR}/bin/zeropad $k 4`
    echo $k
    applywarp --rel --interp=spline -i ${OutputDC}/AP_prevols/vol${vnum}.nii.gz -r ${OutputDC}/SBRefAP_dc.nii.gz -w ${OutputDC}/SBRefAP2FMWarpField.nii.gz -o ${OutputDC}/AP_postvols/vol${k}.nii.gz
    fslmaths ${OutputDC}/AP_postvols/vol${k}.nii.gz -mul ${OutputDC}/Jacobian_01.nii.gz ${OutputDC}/AP_postvols/vol${k}_jac.nii.gz
    FrameMergeDC_AP="${FrameMergeDC_AP}${OutputDC}/AP_postvols/vol${k}_jac.nii.gz "

    k=`echo "$k + 1" | bc`
  done

  echo merge multiple 3D image into one 4D image
  fslmerge -tr ${OutputDC}/AP_st_dc $FrameMergeDC_AP $TR_vol_AP
fi

if [ -f "$PA_st" ]; then
  echo "PA--------->"

  mkdir -p ${OutputDC}/PA_postvols

  TR_vol_PA=`${FSLDIR}/bin/fslval ${PA_st} pixdim4 | cut -d " " -f 1`
  NumFrames_PA=`${FSLDIR}/bin/fslval ${PA_st} dim4`

  FrameMergeDC_PA=""
  k=0
  while [ $k -lt $NumFrames_PA ]; do
    vnum=`${FSLDIR}/bin/zeropad $k 4`
    echo $k
    applywarp --rel --interp=spline -i ${OutputDC}/PA_prevols/vol${vnum}.nii.gz -r ${OutputDC}/SBRefPA_dc.nii.gz -w ${OutputDC}/SBRefPA2FMWarpField.nii.gz -o ${OutputDC}/PA_postvols/vol${k}.nii.gz
    fslmaths ${OutputDC}/PA_postvols/vol${k}.nii.gz -mul ${OutputDC}/Jacobian_02.nii.gz ${OutputDC}/PA_postvols/vol${k}_jac.nii.gz
    FrameMergeDC_PA="${FrameMergeDC_PA}${OutputDC}/PA_postvols/vol${k}_jac.nii.gz "

    k=`echo "$k + 1" | bc`
  done

  echo merge multiple 3D image into one 4D image
  fslmerge -tr ${OutputDC}/PA_st_dc $FrameMergeDC_PA $TR_vol_PA
fi

AP_st_dc=${OutputDC}/AP_st_dc.nii.gz
PA_st_dc=${OutputDC}/PA_st_dc.nii.gz

Predefined_Ref_img=/public_bme2/bme-liyuanning/lishr/Cross_modal/Data/${subject}/func/Ref_img.nii.gz
if [ -f ${Predefined_Ref_img} ]; then
  Ref_img=${Predefined_Ref_img}
else
  cp ${OutputDC}/SBRefAP_dc_jac.nii.gz $Predefined_Ref_img
  Ref_img=${Predefined_Ref_img}
fi

##########头动矫正##########

echo "Head Motion Correction using mcflirt"
echo "########################################"

mkdir -p ${OutputMC}

if [ -f "$AP_st_dc" ]; then
  mcflirt -in ${AP_st_dc} -r ${Ref_img} -o ${OutputMC}/AP_st_dc_mc -mats -plots -report -stages 4 -sinc_final
  python $RootPath/calcul_FD.py ${OutputMC}/AP_st_dc_mc.par ${OutputMC}/PhaseAP_FD.txt  # 计算两帧之差
fi

if [ -f "$PA_st_dc" ]; then
  mcflirt -in ${PA_st_dc} -r ${Ref_img} -o ${OutputMC}/PA_st_dc_mc -mats -plots -report -stages 4 -sinc_final
  python $RootPath/calcul_FD.py ${OutputMC}/PA_st_dc_mc.par ${OutputMC}/PhasePA_FD.txt  # 计算两帧之差
fi

##########将EPI配准到T1##########

echo "register EPI to T1"
date
echo "########################################"

mkdir -p ${OutputSEG}

bet ${T1} ${OutputSEG}/T1_bet -R -B -m -f 0.3
fast -t 1 -n 3 -g ${OutputSEG}/T1_bet

T1_bet="${OutputSEG}/T1_bet.nii.gz"
T1_GM="${OutputSEG}/T1_bet_seg_0.nii.gz"
T1_WM="${OutputSEG}/T1_bet_seg_1.nii.gz"
Tissue="${OutputSEG}/T1_bet_seg.nii.gz"

RefImgforOneStep=${OutputSEG}/T1DownResample.nii.gz
flirt -in ${T1_bet} -ref ${T1_bet} -o ${RefImgforOneStep} -applyisoxfm 2.5 -interp nearestneighbour
fslmaths $RefImgforOneStep -bin "${OutputSEG}/BrainMask"

echo "FLIRT pre-alignment"
date
echo "########################################"

mkdir -p ${OutputF2S}
dof=6

vout=Epi2Str
flirt -ref ${T1_bet} -in ${Ref_img} -dof ${dof} -omat ${OutputF2S}/${vout}_init.mat -out ${OutputF2S}/${vout}_init -cost mutualinfo -searchcost mutualinfo -searchrx -30 30 -searchry -30 30 -searchrz -30 30
# do the second time regisration using bbr
echo "Running BBR"
flirt -ref ${T1_bet} -in ${Ref_img} -dof ${dof} -cost bbr -wmseg ${T1_WM} -init ${OutputF2S}/${vout}_init.mat -applyxfm -omat ${OutputF2S}/${vout}.mat -out ${OutputF2S}/${vout} -schedule ${FSLDIR}/etc/flirtsch/bbr.sch
applywarp -i ${Ref_img} -r ${T1_bet} -o ${OutputF2S}/${vout} --premat=${OutputF2S}/${vout}.mat --interp=spline

echo "one step resampling"
date
echo "########################################"

if [ -f "$PhaseAP" ]; then
  echo "AP---->"
  echo "########################################"

  mkdir -p ${OutputF2S}/AP_warp
  mkdir -p ${OutputF2S}/AP_jacobian
  mkdir -p ${OutputF2S}/AP_postvols

  TR_vol_AP=`${FSLDIR}/bin/fslval ${cutAP} pixdim4 | cut -d " " -f 1`
  NumFrames_AP=`${FSLDIR}/bin/fslval ${cutAP} dim4`

  FrameMergeSTRING_AP=""
  k=0
  while [ $k -lt $NumFrames_AP ] ; do
    vnum=`${FSLDIR}/bin/zeropad $k 4`
    echo $k

    convertwarp --relout --rel --ref=${Ref_img} --warp1=${OutputDC}/SBRefAP2FMWarpField.nii.gz --postmat=${OutputMC}/AP_st_dc_mc.mat/MAT_${vnum} --out=${OutputF2S}/AP_warp/vol${vnum}_temp_warp.nii.gz
    convertwarp --relout --rel --ref=${RefImgforOneStep} --warp1=${OutputF2S}/AP_warp/vol${vnum}_temp_warp.nii.gz --postmat=${OutputF2S}/${vout}.mat --out=${OutputF2S}/AP_warp/vol${vnum}_all_warp.nii.gz

    applywarp --rel --interp=spline --ref=${RefImgforOneStep} -i ${OutputDC}/Jacobian_01.nii.gz --premat=${OutputMC}/AP_st_dc_mc.mat/MAT_${vnum} --postmat=${OutputF2S}/${vout}.mat --out=${OutputF2S}/AP_jacobian/vol${vnum}_Jacobian_01.nii.gz

    applywarp --rel --interp=spline --ref=${RefImgforOneStep} -i ${OutputDC}/AP_prevols/vol${vnum}.nii.gz --warp=${OutputF2S}/AP_warp/vol${vnum}_all_warp.nii.gz --out=${OutputF2S}/AP_postvols/vol${vnum}.nii.gz
    fslmaths ${OutputF2S}/AP_postvols/vol${vnum}.nii.gz -mul ${OutputF2S}/AP_jacobian/vol${vnum}_Jacobian_01.nii.gz ${OutputF2S}/AP_postvols/vol${vnum}_jac.nii.gz

    FrameMergeSTRING_AP="${FrameMergeSTRING_AP}${OutputF2S}/AP_postvols/vol${vnum}_jac.nii.gz "

    k=`echo "$k + 1" | bc`
  done

  echo merge multiple 3D image into one 4D image
  fslmerge -tr ${OutputF2S}/AP_fMRIAfterMinP $FrameMergeSTRING_AP $TR_vol_AP
fi

if [ -f "$PhasePA" ]; then
  echo "PA---->"
  echo "########################################"

  mkdir -p ${OutputF2S}/PA_warp
  mkdir -p ${OutputF2S}/PA_jacobian
  mkdir -p ${OutputF2S}/PA_postvols

  TR_vol_PA=`${FSLDIR}/bin/fslval ${cutPA} pixdim4 | cut -d " " -f 1`
  NumFrames_PA=`${FSLDIR}/bin/fslval ${cutPA} dim4`

  FrameMergeSTRING_PA=""
  k=0
  while [ $k -lt $NumFrames_PA ] ; do
    vnum=`${FSLDIR}/bin/zeropad $k 4`
    echo $k

    convertwarp --relout --rel --ref=${Ref_img} --warp1=${OutputDC}/SBRefPA2FMWarpField.nii.gz --postmat=${OutputMC}/PA_st_dc_mc.mat/MAT_${vnum} --out=${OutputF2S}/PA_warp/vol${vnum}_temp_warp.nii.gz
    convertwarp --relout --rel --ref=${RefImgforOneStep} --warp1=${OutputF2S}/PA_warp/vol${vnum}_temp_warp.nii.gz --postmat=${OutputF2S}/${vout}.mat --out=${OutputF2S}/PA_warp/vol${vnum}_all_warp.nii.gz

    applywarp --rel --interp=spline --ref=${RefImgforOneStep} -i ${OutputDC}/Jacobian_02.nii.gz --premat=${OutputMC}/PA_st_dc_mc.mat/MAT_${vnum} --postmat=${OutputF2S}/${vout}.mat --out=${OutputF2S}/PA_jacobian/vol${vnum}_Jacobian_02.nii.gz

    applywarp --rel --interp=spline --ref=${RefImgforOneStep} -i ${OutputDC}/PA_prevols/vol${vnum}.nii.gz --warp=${OutputF2S}/PA_warp/vol${vnum}_all_warp.nii.gz --out=${OutputF2S}/PA_postvols/vol${vnum}.nii.gz
    fslmaths ${OutputF2S}/PA_postvols/vol${vnum}.nii.gz -mul ${OutputF2S}/PA_jacobian/vol${vnum}_Jacobian_02.nii.gz ${OutputF2S}/PA_postvols/vol${vnum}_jac.nii.gz

    FrameMergeSTRING_PA="${FrameMergeSTRING_PA}${OutputF2S}/PA_postvols/vol${vnum}_jac.nii.gz "

    k=`echo "$k + 1" | bc`
  done

  echo merge multiple 3D image into one 4D image
  fslmerge -tr ${OutputF2S}/PA_fMRIAfterMinP $FrameMergeSTRING_PA $TR_vol_PA
fi

AP_minP=${OutputF2S}/AP_fMRIAfterMinP.nii.gz
PA_minP=${OutputF2S}/PA_fMRIAfterMinP.nii.gz

echo "show some GIF figure for check"
date
echo "########################################"
mkdir -p ${OutputFIG}

python $RootPath/generate_info.py --path=$ResultsPath --rootpath=$RootPath
python $RootPath/show_fmri.py --data=$AP_minP --out=${OutputFIG}/Phase_AP
python $RootPath/show_fmri.py --data=$PA_minP --out=${OutputFIG}/Phase_PA

echo "Complete Volume Preprocess"
date

echo "Resample the data onto the surface"
date
echo "########################################"

mkdir -p ${OutputSURF}

mri_vol2surf --src ${AP_minP} --out ${OutputSURF}/AP_lh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi lh
mri_vol2surf --src ${AP_minP} --out ${OutputSURF}/AP_rh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi rh
mri_vol2surf --src ${PA_minP} --out ${OutputSURF}/PA_lh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi lh
mri_vol2surf --src ${PA_minP} --out ${OutputSURF}/PA_rh_surf.mgh --regheader ${subject_name} --projfrac-avg 0 1 0.2 --interp trilinear --hemi rh

echo "Complete Surface Preprocess"
date