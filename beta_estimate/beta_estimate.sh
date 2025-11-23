module load apps/matlab/2021b

cd /public/home/lishr2022/Project/Cross-modal/beta_estimate

subject=$1
#python step1_rescale.py
#matlab -nodisplay -r "step2_beta_estimate('${subject}'); exit;"
#matlab -nodisplay -r "run('test_onoff'); exit;"

#python unmatch_step1_rescale.py
#matlab -nodisplay -r "unmatch_step2_beta_estimate('${subject}'); exit;"
#python unmatch_step3_analysis.py
#matlab -nodisplay -r "run('unmatch_step4_onoff'); exit;"

matlab -nodisplay -r "volume_step2_beta_estimate('${subject}'); exit;"