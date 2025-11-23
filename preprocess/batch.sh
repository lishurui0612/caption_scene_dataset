Subject_name=$1
Root=$2
T1=$3

for file in "$Root"/*
do

  result=$(echo $file | grep "fLoc")
  if [ "$result" != "" ]; then
    for t in "$file"/*
    do
      echo $t
      echo "###########################"
      lines=$(find $t -name "*SURF*" | wc -l)
      if [ $lines -eq 0 ]; then
        PhaseAP=$(find $t -name "*AP_2.5mm*.nii.gz")
        PhasePA=$(find $t -name "*PA_2.5mm*.nii.gz")
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $t $T1 $PhaseAP $PhasePA 0 156 $Subject_name
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $t
      fi
    done
  fi

  result=$(echo $file | grep "retinotopic_93_multibar")
  if [ "$result" != "" ]; then
    echo $file
    echo "###########################"
    lines=$(find $file -name "*SURF*" | wc -l)
    if [ $lines -eq 0 ]; then
      PhaseAP=$(find $file -name "*AP_2.5mm*.nii.gz")
      PhasePA=$(find $file -name "*PA_2.5mm*.nii.gz")
      bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $file $T1 $PhaseAP $PhasePA 0 150 $Subject_name
      bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $file
    fi
  fi

  result=$(echo $file | grep "retinotopic_94_wedgeringmash")
  if [ "$result" != "" ]; then
    echo $file
    echo "###########################"
    lines=$(find $file -name "*SURF*" | wc -l)
    if [ $lines -eq 0 ]; then
      PhaseAP=$(find $file -name "*AP_2.5mm*.nii.gz")
      PhasePA=$(find $file -name "*PA_2.5mm*.nii.gz")
      bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $file $T1 $PhaseAP $PhasePA 0 150 $Subject_name
      bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $file
    fi
  fi

  result=$(echo $file | grep "stimulus")
  if [ "$result" != "" ]; then
    for t in "$file"/*
    do
      echo $t
      echo "###########################"
      lines=$(find $t -name "*SURF*" | wc -l)
      if [ $lines -eq 0 ]; then
        PhaseAP=$(find $t -name "*AP_2.5mm*.nii.gz")
        PhasePA=$(find $t -name "*PA_2.5mm*.nii.gz")
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $t $T1 $PhaseAP $PhasePA 0 150 $Subject_name
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $t
      fi
    done
  fi

  result=$(echo $file | grep "onlyimage")
  if [ "$result" != "" ]; then
    for t in "$file"/*
    do
      echo $t
      echo "###########################"
      lines=$(find $t -name "*SURF*" | wc -l)
      if [ $lines -eq 0 ]; then
        PhaseAP=$(find $t -name "*AP_2.5mm*.nii.gz")
        PhasePA=$(find $t -name "*PA_2.5mm*.nii.gz")
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $t $T1 $PhaseAP $PhasePA 0 150 $Subject_name
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $t
      fi
    done
  fi

  result=$(echo $file | grep "winoground")
  if [ "$result" != "" ]; then
    for t in "$file"/*
    do
      echo $t
      echo "###########################"
      lines=$(find $t -name "*SURF*" | wc -l)
      if [ $lines -eq 0 ]; then
        PhaseAP=$(find $t -name "*AP_2.5mm*.nii.gz")
        PhasePA=$(find $t -name "*PA_2.5mm*.nii.gz")
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $t $T1 $PhaseAP $PhasePA 0 150 $Subject_name
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $t
      fi
    done
  fi

  result=$(echo $file | grep "unmatch")
  if [ "$result" != "" ]; then
    for t in "$file"/*
    do
      echo $t
      echo "###########################"
      lines=$(find $t -name "*SURF*" | wc -l)
      if [ $lines -eq 0 ]; then
        PhaseAP=$(find $t -name "*AP_2.5mm*.nii.gz")
        PhasePA=$(find $t -name "*PA_2.5mm*.nii.gz")
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S3_preprocess_new.sh $t $T1 $PhaseAP $PhasePA 0 156 $Subject_name
        bash /public/home/lishr2022/Project/Cross-modal/Preprocess/S4_remove_file.sh $t
      fi
    done
  fi

done