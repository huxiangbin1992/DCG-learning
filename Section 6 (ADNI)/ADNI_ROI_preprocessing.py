# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:21:00 2026

@author: 16847
"""

import os
import numpy as np
import ants
import pandas as pd

tpl_t1   = ants.image_read("T_template0.nii.gz")  # OASIS T1 template
tpl_mb = ants.image_read("OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz") # MindBoggle-101 Atlas template

###### You need to download the above two templates

bins = 50
t_grid = (np.arange(bins) + 0.5) / bins
lab = tpl_mb.numpy().astype(int) # label of the tramsformed parcellation
labels = np.unique(lab[(lab > 0)]) # non zero label list in brain
num_roi = len(labels)
ROI_center_mat = np.zeros((num_roi, 3))

for k in range(num_roi):
    rid = labels[k]
    pts = np.argwhere(lab == rid).astype(float)
    c = pts.mean(axis=0) # geometric median: minimize sum ||pts_i - c||
    for _ in range(200):
        d = np.linalg.norm(pts - c, axis=1)
        w = 1.0 / np.maximum(d, 1e-12)
        c_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(c_new - c) < 1e-5:
            break
        c = c_new
    ROI_center_mat[k, :] = c
c_min = ROI_center_mat.min(axis=0)
c_max = ROI_center_mat.max(axis=0)
ROI_center_mat = (ROI_center_mat - c_min) / (c_max - c_min)

base_dir = r"ADNI3"
out_dir = r"out_ADNI3"
os.makedirs(out_dir, exist_ok=False)
def pick_one_image_folder(patient_dir):
    candidates = []
    for root, dirs, files in os.walk(patient_dir):
        nii_files = [f for f in files if f.endswith(".nii") or f.endswith(".nii.gz")]
        if nii_files and os.path.basename(root).startswith("I"):
            date_time = os.path.basename(os.path.dirname(root))
            candidates.append((date_time, root, nii_files[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1])) 
    return candidates[-1] 

for patient_id in sorted(os.listdir(base_dir)):
    patient_dir = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_dir):
        continue
    pick = pick_one_image_folder(patient_dir)
    if pick is None:
        continue
    date_time, img_dir, nii_name = pick
    nii_path = os.path.join(img_dir, nii_name)
    img_id = os.path.basename(img_dir)    
    img_raw = ants.image_read(nii_path)

    # ========================= 
    # Step 1) N4 bias correction 
    # ========================= 
    subj_n4  = ants.n4_bias_field_correction(img_raw) 
    reg = ants.registration(fixed=tpl_t1, moving=subj_n4, type_of_transform="SyN")  #registration 

    # ========================= 
    # Step 2) registration-based brain extraction 
    # Step 3) prior-based N4-Atropos 6 tissue segmentation 
    # ========================= 
    ############# Useless in calculation of ROI ############### 

    # ========================= 
    # Step 4) multi-atlas cortical parcellation by MindBoggle-101 dataset 
    # ========================= 
    warp_tx = [t for t in reg["fwdtransforms"] if "Warp" in t][0] # SyN registration fwdtransforms
    logjac_tpl = ants.create_jacobian_determinant_image(domain_image=tpl_t1, tx=warp_tx, do_log=True) # log-Jacobian
    J = logjac_tpl.numpy()
    finite_m = np.isfinite(J)   
    J_vmin, J_vmax = np.quantile(J[finite_m], [0.0001, 0.9999]) # normal range of log-Jacobian

    # ========================= 
    # Step 5) Volume density curves of ROIs 
    # ========================= 
    # function for ROI density curve: histogram density on voxel-wise Jacobian values within each ROI 
    edges = np.linspace(J_vmin, J_vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma_bins = 1.0 
    radius = int(3 * sigma_bins)
    xk = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xk / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()
    bin_width = edges[1] - edges[0]

    lqdt_dict = np.zeros((num_roi, bins))
    count_dict = np.zeros(num_roi)

    for k in range(num_roi):
        rid = labels[k]
        v = J[(lab == rid) & finite_m]
        v = v[(v >= J_vmin) & (v <= J_vmax)]
        count_dict[k] = v.size
        if v.size != 0:
            hist, _ = np.histogram(v, bins=bins, range=(J_vmin, J_vmax), density=True) # f(x): histogram density on fixed grid
            # --- smooth histogram ---
            hist = np.convolve(hist, kernel, mode="same")
            hist = hist / (hist.sum() * bin_width)
            Q = np.quantile(v, t_grid) # Q(t): sample quantiles
            f_at_Q = np.interp(Q, centers, hist) # f(Q(t)): interpolate density at quantiles
            lqdt_dict[k, :] = -np.log(np.maximum(f_at_Q, 1e-10)) # log quantile density transform

    out_A = os.path.join(out_dir, f"{patient_id}__{img_id}__lqdt.csv")
    pd.DataFrame(lqdt_dict).to_csv(out_A, index=False, header=False)

pd.DataFrame(ROI_center_mat).to_excel("ROI_center_mat.xlsx", index=False, header=False)