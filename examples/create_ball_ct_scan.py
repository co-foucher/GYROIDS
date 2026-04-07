"""
Example: Generate a synthetic CT scan of a ball (20x20x20 voxels) and save as .mhd
"""

import numpy as np
import SimpleITK as sitk

# --- Parameters ---
resolution = 20          # voxels per axis
spacing    = (1.0, 1.0, 1.0)   # mm per voxel
output_path = "ball_ct_scan.mhd"

# --- Build 3-D boolean ball ---
center = (resolution - 1) / 2.0
radius = center * 0.8          # 80 % of the half-extent

indices = np.arange(resolution)
z, y, x = np.meshgrid(indices, indices, indices, indexing="ij")  # shape (20,20,20)

distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)

# Foreground (ball) = 255, background = 0
volume = np.where(distance <= radius, np.uint8(255), np.uint8(0))

# --- Convert to SimpleITK image and save ---
sitk_image = sitk.GetImageFromArray(volume)
sitk_image.SetSpacing(spacing)
sitk_image.SetOrigin((0.0, 0.0, 0.0))

sitk.WriteImage(sitk_image, output_path)
print(f"Saved CT scan to: {output_path}  (shape {volume.shape}, dtype {volume.dtype})")
