"""
Free Water DTI Processing Script for Substantia Nigra (SN)

This script processes 7T diffusion MRI data for a list of subjects to estimate free water-corrected DTI metrics
within the Substantia Nigra (SN), using a bounding box approach. It performs the following steps for each subject:

1. Loads preprocessed DTI data and brain masks.
2. Loads left and right SN ROI masks (from the PAULI atlas or equivalent).
3. Computes a 3-voxel margin bounding box around the bilateral SN mask.
4. Crops the DTI data and mask to this bounding box for efficient fitting.
5. Fits the Free Water DTI model (using Dipy's fwdti implementation).
6. Extracts FA, MD, and free water fraction (FW) maps.
7. Saves the full-size SN-cropped FA, MD, and FW maps for each subject.
8. Computes and logs average FA, MD, and FW values in left and right SN.
9. Saves summary metrics for all subjects into a single CSV file.
10. Logs any failures with error messages.

Inputs:
- Raw DTI data directory (`base_raw_dti`)
- Preprocessed DTI data directory (`base_preproc_dti`)
- ROI mask directory with PAULI SN masks (`base_masks`)
- Subject list text file (`subject_list_file`)

Outputs:
- FA, MD, FW NIfTI maps saved in `base_deriv/{subject}/`
- Summary CSV at `output_csv` with mean metrics for each subject
- Log of failures at `log_file`

Dependencies:
- nibabel, numpy, pandas
- Dipy (for Free Water DTI)
- All required NIfTI files must exist and be properly aligned

Author: [Laura Neville]
Date: [26.05.2025]
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.fwdti as fwdti
import concurrent.futures

# Paths
# Here my mask files and data files are in different directories - change if this is not the case
base_raw_dti       = '/media/laura/backup/7T_DTI_RAW'
base_preproc_dti   = '/media/laura/backup/7T_preprocessed_HCP_DTI'
base_masks         = '/media/laura/data/SN_7T_sleep_analysis_ATPPy/7T_HCP_output_sleep_py'
base_deriv = '/media/laura/data/SN_7T_sleep_analysis_ATPPy/derivatives/fwDTI'
subject_list_file  = os.path.join(base_raw_dti, 'subject_list_7T_RAW_n175.txt')
output_csv   = os.path.join(base_deriv, 'SN_fwDTI_results.csv')
log_file     = os.path.join(base_deriv, 'SN_fwDTI_failures.log')

# Load subject list
with open(subject_list_file, 'r') as f:
    subjects = [s.strip() for s in f if s.strip()]

def process_subject(subj):
    """Process one subject: fits FW-DTI in SN bbox, returns stats or raises."""
    print(f"\n🚀 Processing subject: {subj}")

    raw_dti       = os.path.join(base_raw_dti, subj, 'Diffusion_7T')
    preproc_dti   = os.path.join(base_preproc_dti, subj, 'Diffusion.bedpostX')
    mask_l_path   = os.path.join(base_masks, subj, 'ROI_masks', f'{subj}_PAULI_SN_L.nii.gz')
    mask_r_path   = os.path.join(base_masks, subj, 'ROI_masks', f'{subj}_PAULI_SN_R.nii.gz')
    save_dir      = os.path.join(base_deriv, f'{subj}')
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    img   = nib.load(os.path.join(raw_dti, 'data.nii.gz'))
    data  = np.asarray(img.dataobj)
    bvals = f'{raw_dti}/bvals'
    bvecs = f'{raw_dti}/bvecs'
    gtab = gradient_table(bvals=bvals, bvecs=bvecs, b0_threshold=70)
    mask  = nib.load(os.path.join(preproc_dti, 'nodif_brain_mask.nii.gz')).get_fdata()

    # Load SN masks
    sn_l = nib.load(mask_l_path).get_fdata() > 0
    sn_r = nib.load(mask_r_path).get_fdata() > 0
    sn_mask = sn_l | sn_r

    # Compute bounding box + margin
    coords = np.array(np.where(sn_mask))
    margin = 3
    x0, x1 = coords[0].min(), coords[0].max() + 1
    y0, y1 = coords[1].min(), coords[1].max() + 1
    z0, z1 = coords[2].min(), coords[2].max() + 1
    x0, x1 = max(x0-margin,0), min(x1+margin,sn_mask.shape[0])
    y0, y1 = max(y0-margin,0), min(y1+margin,sn_mask.shape[1])
    z0, z1 = max(z0-margin,0), min(z1+margin,sn_mask.shape[2])

    # Crop
    data_crop = data[x0:x1, y0:y1, z0:z1, :]
    mask_crop = mask[x0:x1, y0:y1, z0:z1]
    sn_l_crop = sn_l[x0:x1, y0:y1, z0:z1]
    sn_r_crop = sn_r[x0:x1, y0:y1, z0:z1]

    # Fit FW-DTI
    fwdtimodel  = fwdti.FreeWaterTensorModel(gtab)
    fwdtifit    = fwdtimodel.fit(data_crop, mask=mask_crop)
    FA = fwdtifit.fa
    MD = fwdtifit.md
    FW = fwdtifit.f

    # Extract statistics
    fa_l = FA[sn_l_crop]
    fa_r = FA[sn_r_crop]
    md_l = MD[sn_l_crop]
    md_r = MD[sn_r_crop]
    fw_l = FW[sn_l_crop]
    fw_r = FW[sn_r_crop]

    # Save full-size volumes
    full_shape = mask.shape
    full_FA    = np.zeros(full_shape)
    full_MD = np.zeros(full_shape)
    full_FW    = np.zeros(full_shape)
    full_FA[x0:x1, y0:y1, z0:z1] = FA
    full_MD[x0:x1, y0:y1, z0:z1] = MD
    full_FW[x0:x1, y0:y1, z0:z1] = FW

    affine = img.affine
    nib.save(nib.Nifti1Image(full_FA, affine),
             os.path.join(save_dir, f'{subj}_fwDTI_FA_SNcrop.nii.gz'))
    nib.save(nib.Nifti1Image(full_MD, affine),
             os.path.join(save_dir, f'{subj}_fwDTI_MD_SNcrop.nii.gz'))
    nib.save(nib.Nifti1Image(full_FW, affine),
             os.path.join(save_dir, f'{subj}_fwDTI_FWfrac_SNcrop.nii.gz'))

    # Return summary
    return {
        'subject': subj,
        'FA_L': np.mean(fa_l),
        'FA_R': np.mean(fa_r),
        'MD_L': np.mean(md_l),
        'MD_R': np.mean(md_r),
        'FW_L': np.mean(fw_l),
        'FW_R': np.mean(fw_r)
    }

# Run sequentially
results = []
with open(log_file, 'w') as logf:
    for subj in subjects:
        try:
            res = process_subject(subj)
            print(f"[✅] {subj} done")
            results.append(res)
        except Exception as e:
            err = str(e).replace('\n', ' ')
            logf.write(f"{subj}: {err}\n")
            print(f"[❌] {subj} failed: {err}")

# Save CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, float_format='%.6f')
print(f"\n📁 Results -> {output_csv}")

