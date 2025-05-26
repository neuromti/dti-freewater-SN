# 🧠 Free Water Imaging with DIPY

This folder contains a full pipeline for Free Water Elimination Diffusion Tensor Imaging (fwDTI) using the DIPY library, based on the official DIPY implementation:
🔗 [DIPY: Free Water DTI example](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_fwdti.html#footcite-pasternak2009)

---

## 📌 Pipeline Overview

The goal of this pipeline is to:
- Estimate **free water volume fraction**
- Generate **free water-corrected FA and MD maps**
- Compare these metrics to standard DTI measures

By accounting for extracellular free water (e.g., from CSF), this model improves the specificity of diffusion metrics to underlying tissue microstructure — especially useful in deep brain regions like the Substantia Nigra.

---

## 🧪 Method Summary

### Inputs:
- Preprocessed 7T diffusion-weighted images (HCP-style)
- Brain mask (`nodif_brain_mask.nii.gz`)
- Subject-specific ROI masks (e.g., PAULI atlas SN left/right masks)
- Gradient table with two b-shells (e.g., b=1000 and b=2000), plus b0 images

### Model:
- **Free Water DTI**: Two-compartment model (tissue + isotropic free water)
- Implemented via `dipy.reconst.fwdti.FreeWaterTensorModel`

### Outputs:
- Full-volume fwDTI maps:
  - `*_fwDTI_FA_SNcrop.nii.gz`
  - `*_fwDTI_MD_SNcrop.nii.gz`
  - `*_fwDTI_FWfrac_SNcrop.nii.gz`
- Summary CSV with average FA, MD, FW in left and right SN
- Visual comparison plots (for testing on one slice)

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `process_fwDTI_pipeline.py` | Full pipeline: loads data, fits model, crops to SN, saves volumes + summary |
| `visualize_fwDTI_slice_comparison.py` | Script to compare FW-DTI vs. standard DTI on a single slice |
| `subject_list_7T_RAW_n175.txt` | List of subjects to process |
| `SN_fwDTI_results.csv` | Output summary with FA/MD/FW per subject (auto-generated) |
| `SN_fwDTI_failures.log` | Log of failed subjects and reasons (auto-generated) |

---

## 🖼️ Visualization

The visualization script shows:

- FW-DTI vs standard DTI FA and MD
- Difference maps
- Free water volume map
- Corresponding brain mask

This is helpful for QC and debugging.

---

## ⚙️ Requirements

- Python 3.8+
- DIPY
- nibabel
- numpy
- pandas
- matplotlib

To install requirements:

bash
pip install dipy nibabel numpy pandas matplotlib

## 📚 References

Pasternak et al., 2009
Free water elimination and mapping from diffusion MRI.
Magnetic Resonance in Medicine, 62(3):717–730.
doi:10.1002/mrm.22055

Hoy et al., 2014
Optimization of a free water elimination two-compartment model for diffusion tensor imaging.
NeuroImage, 103:323–333.
doi:10.1016/j.neuroimage.2014.09.053

Neto Henriques et al., 2017
[Re] Optimization of a free water elimination two-compartment model for diffusion tensor imaging.
bioRxiv.
doi:10.1101/108795
