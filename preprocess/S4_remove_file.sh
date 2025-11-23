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
OutputDC="${ResultsPath}/DC"
OutputMC="${ResultsPath}/MC"
OutputSEG="${ResultsPath}/SEG"
OutputF2S="${ResultsPath}/F2S"
OutputFL="${ResultsPath}/FL"

AP_MinP=$OutputF2S/AP_fMRIAfterMinP.nii.gz
PA_MinP=$OutputF2S/PA_fMRIAfterMinP.nii.gz

AP_MC=$OutputMC/AP_st_dc_mc.par
PA_MC=$OutputMC/PA_st_dc_mc.par

cutAP=${ResultsPath}/cutAP.nii.gz
cutPA=${ResultsPath}/cutPA.nii.gz
SBRefAP=${ResultsPath}/SBRefAP.nii.gz
SBRefPA=${ResultsPath}/SBRefPA.nii.gz
AP_st=${ResultsPath}/AP_st.nii.gz
PA_st=${ResultsPath}/PA_st.nii.gz

echo $ResultsPath
echo "Remove several outputs"
echo "########################################"
cp $AP_MinP $ResultsPath
cp $PA_MinP $ResultsPath

cp $AP_MC $ResultsPath
cp $PA_MC $ResultsPath

if [ -d $OutputDC ]; then
  rm -rf $OutputDC
fi

if [ -d $OutputMC ]; then
  rm -rf $OutputMC
fi

if [ -d $OutputSEG ]; then
  rm -rf $OutputSEG
fi

if [ -d $OutputF2S ]; then
  rm -rf $OutputF2S
fi

if [ -d $OutputFL ]; then
  rm -rf $OutputFL
fi

if [ -f $cutAP ]; then
  rm -rf $cutAP
fi

if [ -f $cutPA ]; then
  rm -rf $cutPA
fi

if [ -f $SBRefAP ]; then
  rm -rf $SBRefAP
fi

if [ -f $SBRefPA ]; then
  rm -rf $SBRefPA
fi

if [ -f $AP_st ]; then
  rm -rf $AP_st
fi

if [ -f $PA_st ]; then
  rm -rf $PA_st
fi