# A large-scale vision-language fMRI dataset for multi-modal semantic processing

---
## About
This repository provides codes for our large-scale fMRI dataset, focusing on four core components:
* **Dataset**: Access the full dataset on [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=3608457c9c5e4c078ecca153c3413b1b).
* **DOI**: https://doi.org/10.57760/sciencedb.27580.
* **Preprocessing**: Pipelines for fMRI data in the `./preprocess/` folder.
* **Quality Control**: Tools for assessing preprocessing and data reliability in the `./quality_control/` folder.
* **Beta Estimation**: Using GLMSingle in the `./beta_estimate/` folder.
* **Encoding Models**: Voxel-wise encoding model code for caption stimuli in the `./encoding/` folder.

---
## Experimental paradigm
Participants performed a text–image semantic matching task in which captions and images were presented in an alternating sequence, and participants judged whether each pair conveyed the same meaning.
<p align="center">
  <img src="assets/paradigm.png" width="700">
</p>

---
## Preprocessing pipeline
Below is the overview of preprocessing pipeline. The detailed procedures and implementation scripts are available in the `./preprocess/` folder.
<p align="center">
  <img src="assets/preprocess.png" width="700">
</p>

---
## Quality control
Below is the QC overview demo showing cross-session alignment results for participant sub-07. The detailed QC procedures and implementation scripts are available in the `./quality_control/` folder.
<p align="center">
  <img src="assets/S7_register.gif" width="700">
</p>

---
## Citation
If you use this dataset or the accompanying code, please cite the following abstract presented in Journal of Vision:

Li, Y., Li, S., Jin, Z., Gu, S., & Zhang, R. Y. (2025). A large-scale vision-language fMRI dataset for multi-modal semantic processing. Journal of Vision, 25(9), 2445–2445.

### **BibTeX**
```bibtex
@article{li2025large,
  title={A large-scale vision-language fMRI dataset for multi-modal semantic processing},
  author={Li, Yuanning and Li, Shurui and Jin, Zheyu and Gu, Shi and Zhang, Ru-Yuan},
  journal={Journal of Vision},
  volume={25},
  number={9},
  pages={2445--2445},
  year={2025},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

---
## Acknowledgements
This work was supported by the **National Science and Technology Major Project of China** (Grant No. 2025ZD0217000 to Y.L.), the **National Natural Science Foundation of China** (Grant No. 32371154 to Y.L.), the **Shanghai Rising-Star Program** (Grant No. 24QA2705500 to Y.L.), and the **Lingang Laboratory** (Grant No. LG-GG-202402-06 to Y.L.).

We would like to express our sincere gratitude to [**Yu-Qi You**](https://orcid.org/0009-0005-5723-1947) for the insightful discussions and support.

---
## License
This repository is released under the **CC BY-NC-ND 4.0** license.  
You may use the code for research and educational purposes, but commercial use, distribution of modified versions, or sublicensing are not permitted.


For full terms, please refer to the [LICENSE](./LICENSE) file.
