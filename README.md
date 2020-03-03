**Denoising Low Dose SPECT-MPI via Convolutional Denoising Networks**

* This repository contains some of the code developed for the study presented in:

[1] A. Juan Ramon, et al., “Improving Diagnostic Accuracy in Low Dose SPECT Myocardial Perfusion Imaging with Convolutional Denoising Networks,” IEEE Trans. Med. Imag. 2020.

Abstract— Lowering the administered dose in SPECT myocardial perfusion imaging (MPI) has become an important clinical problem. In this study we investigate the potential benefit of applying a deep learning (DL) approach for suppressing the elevated imaging noise in low-dose SPECT-MPI studies. We adopt a supervised learning approach to train a neural network by using image pairs obtained from full-dose (target) and low-dose (input) acquisitions of the same patients. In the experiments, we made use of acquisitions from 1,052 subjects and demonstrate the approach for two commonly used reconstruction methods in clinical SPECT-MPI: 1) filtered backprojection (FBP), and 2) ordered-subsets expectation-maximization (OSEM) with corrections for attenuation, scatter and resolution. We evaluated the DL output for the clinical task of perfusion-defect detection at
a number of successively reduced dose levels (1/2, 1/4, 1/8, 1/16 of full dose). The results indicate that the proposed DL approach can achieve substantial noise reduction and lead to improvement in the diagnostic accuracy of low-dose data. In particular, at 1⁄2 dose, DL yielded an area-under-the-ROC-curve (AUC) of 0.799, which is nearly identical to the AUC=0.801 obtained by OSEM at full-dose (p-value=0.73); similar results were also obtained for FBP reconstruction. Moreover, even at 1/8 dose, DL achieved AUC=0.770 for OSEM, which is above the AUC=0.755 obtained at full-dose by FBP. These results indicate that, compared to conventional reconstruction filtering, DL denoising can allow for additional dose reduction without sacrificing the diagnostic accuracy in SPECT-MPI.

---

## List of scripts:
1. `build_NN.py` contains functions to define several network structures tested in the study.
2. `test_CAE_3D.py` contains main script for testing trained networks and saving results

---
