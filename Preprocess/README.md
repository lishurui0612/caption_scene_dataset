# Cross-modal-Processing-pipeline
**The processing pipeline of cross-modal fMRI data***


```angular2html
ResultsPath="/public/home/lishr2022/Project/Cross-modal/test_data"
vol_preprocess.sh $ResultsPath
surf_preprocess.sh $ResultsPath
```
ResultsPath里需要包含PhaseAP、PhasePA、T1w的图像 

vol_preprocess.sh执行完之后最终的文件在${ResultsPath}/FL/AP_fMRIAfterfilter.nii.gz和${ResultsPath}/FL/PA_fMRIAfterfilter.nii.gz里

surf_preprocess.sh执行完后结果在${ResultsPath}/SURF里

**vol_preprocess.sh中包含：** 
1. Slice Timing Correction 
2. Select the mean Image as SBRef image 
3. EPI Distortion Correction using topup 
4. Head Motion Correction using mcflirt 
5. Register EPI to T1 
6. One step resampling 
7. High Pass Filter 
8. Generate some GIFs/Images for check (including topup、MC)

**surf_preprocess.sh中包含：** 
1. Recon all by FreeSurfer 
2. Boundary-based Register by FSL in FreeSurfer \
3. Resample data onto the surface