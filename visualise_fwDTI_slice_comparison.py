"""
Visual Comparison of Free Water DTI vs Standard DTI on a Midbrain Slice

This script performs a slice-based visualization of Free Water DTI (FW-DTI) and standard DTI fits
for a single subject. It allows quick inspection of modeling quality, differences in FA/MD, and
free water volume in a selected axial slice.

Steps:
1. Loads raw DTI data, nodif brain mask, and gradient table
2. Selects an axial slice (default = 60) for inspection
3. Fits FW-DTI and standard DTI models to the slice
4. Extracts and plots:
   - FA (FW-DTI vs standard DTI)
   - MD (FW-DTI vs standard DTI)
   - Differences in FA/MD
   - Free water volume map (f)
   - The corresponding brain mask

This script is useful for:
- Verifying model fit visually
- Comparing FW-DTI corrections vs standard DTI
- Sanity-checking spatial alignment and signal quality

Adjust:
- `slice_idx` to visualize a different region
- Subject paths or add CLI args if needed

Author: [Laura Neville]
Date: [26.05.2025]
"""


import os.path as op
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import fetch_hbn
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti

# Paths
raw_dti = '/media/laura/backup/7T_DTI_RAW/100610/Diffusion_7T'
preproc_dti = '/media/laura/backup/7T_preprocessed_HCP_DTI/100610/Diffusion.bedpostX'

# Load data
img = nib.load(f'{raw_dti}/data.nii.gz')
data = data = np.asarray(img.dataobj)

mask = nib.load(f'{preproc_dti}/nodif_brain_mask.nii.gz').get_fdata()
bvals = f'{raw_dti}/bvals'
bvecs = f'{raw_dti}/bvecs'
gtab = gradient_table(bvals=bvals, bvecs=bvecs, b0_threshold=70)

print("Mask shape:", mask.shape)
print("Sum of mask in slice:", np.sum(mask[:, :, 60:61]))

print("Nonzero mask voxels:", np.sum(mask > 0))


# plt.imshow(mask[:, :,60].T, origin='lower', cmap='gray')
# plt.title("Brain mask at slice 60")
# plt.show()

# for z in range(mask.shape[2]):
#     nonzero_voxels = np.sum(mask[:, :, z] > 0)
#     if nonzero_voxels > 0:
#         print(f"Slice {z} has {nonzero_voxels} voxels in mask")


# #Subset of data
# data_small = data[:, :, 60:61]
# mask_small = mask[:, :, 60:61]
slice_idx = 60

# print("Mask shape:", mask_small.shape)

# Slice and squeeze to remove Z dimension
data_small = np.squeeze(data[:, :, slice_idx:slice_idx+1, :])   # → (173, 207, N)
mask_small = np.squeeze(mask[:, :, slice_idx:slice_idx+1])      # → (173, 207)

# Confirm shape
print("Data shape:", data_small.shape)  # should be (173, 207, N)
print("Mask shape:", mask_small.shape)  # should be (173, 207)

# FreeWaterTensorModel class object
fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

#Fit
fwdtifit = fwdtimodel.fit(data_small, mask=mask_small)
# fwdtifit = fwdtimodel.fit(data, mask=mask)

#Extract FA and MD
FA = fwdtifit.fa
MD = fwdtifit.md
F = fwdtifit.f

# Remove very high FA values
FA[FA > 0.7] = 0

print("FA shape:", FA.shape)
print("FW FA min/max:", np.nanmin(FA), np.nanmax(FA))
print("FW FA unique values:", np.unique(FA[~np.isnan(FA)]))

# Check FW FA/MD values
print("FW FA range:", np.nanmin(FA), np.nanmax(FA))
print("FW MD range:", np.nanmin(MD), np.nanmax(MD))

# Check free water fraction (f)
print("Free water volume range:", np.nanmin(F), np.nanmax(F))

#Comparison to standard measures
dtimodel = dti.TensorModel(gtab)

dtifit = dtimodel.fit(data_small, mask=mask_small)
# dtifit = dtimodel.fit(data, mask=mask)

dti_FA = dtifit.fa
dti_MD = dtifit.md

#Remove very high FA values
dti_FA[dti_FA > 0.7] = 0


fig1, axs = plt.subplots(1, 8, figsize=(16, 6), subplot_kw={"xticks": [], "yticks": []})
fig1.subplots_adjust(hspace=0.3, wspace=0.05)

axs[0].imshow(FA.T, origin='lower', cmap='gray', vmin=0, vmax=1)
axs[0].set_title("A) FW DTI FA")

axs[1].imshow(dti_FA.T, origin='lower', cmap='gray')
axs[1].set_title("B) standard DTI FA")

FAdiff = abs(FA - dti_FA)
axs[2].imshow(FAdiff.T, origin='lower', cmap='gray')
axs[2].set_title("C) FA difference")

axs[3].imshow(MD.T, origin='lower', cmap='gray', vmin=0, vmax=2.5e-3)
axs[3].set_title("D) FW DTI MD")

axs[4].imshow(dti_MD.T, origin='lower', cmap='gray')
axs[4].set_title("E) standard DTI MD")

MDdiff = abs(MD - dti_MD)
axs[5].imshow(MDdiff.T, origin='lower', cmap='gray')
axs[5].set_title("F) MD difference")

axs[6].imshow(F.T, origin='lower', cmap='gray', vmin=0, vmax=1)
axs[6].set_title("G) Free water volume")

axs[7].imshow(mask_small.T, origin='lower', cmap='gray')
axs[7].set_title("H) Mask")

plt.tight_layout()
plt.show()
fig1.savefig("In_vivo_free_water_DTI_and_standard_DTI_measures.png")

