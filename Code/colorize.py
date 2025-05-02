#!/usr/bin/env python3
import os, sys
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIGURATION — just edit these two values and re-run:
# ------------------------------------------------------------
d       = 1        # neighbourhood radius: 1 ⇒ 3×3, 2 ⇒ 5×5, 3 ⇒ 7×7, etc.
REG_EPS = 1e-6     # ε added to variance: try 1e-2, 1e-1, 1e0 to force more bleed

# (Optional extra knob — scale local variance up/down:)
ALPHA   = 1.0      # sigma2 = ALPHA*var + REG_EPS

# ------------------------------------------------------------
# Directories (preserved from original structure)
# ------------------------------------------------------------
GRAY_DIR     = os.path.join("..", "Gray_Scale")
SCRIBBLE_DIR = os.path.join("..", "Scribble")
RESULT_DIR   = os.path.join("..", "Results")

# ------------------------------------------------------------
# CLI usage: exactly 3 args, no flags
# ------------------------------------------------------------
if len(sys.argv) != 4:
    print("Usage: python colorize.py <gray_filename> <scribble_filename> <result_filename>")
    sys.exit(1)
gray_name, scribble_name, result_name = sys.argv[1:]

# build paths
gray_path     = os.path.join(GRAY_DIR,     gray_name)
scribble_path = os.path.join(SCRIBBLE_DIR, scribble_name)
result_path   = os.path.join(RESULT_DIR,   result_name)

# ------------------------------------------------------------
# load & validate
# ------------------------------------------------------------
gray_img     = cv2.imread(gray_path,     cv2.IMREAD_GRAYSCALE)
scribble_img = cv2.imread(scribble_path, cv2.IMREAD_COLOR)
if gray_img is None or scribble_img is None:
    print("Error loading images; check paths.")
    sys.exit(1)
h, w = gray_img.shape
if scribble_img.shape[:2] != (h, w):
    print("Error: size mismatch between gray and scribble.")
    sys.exit(1)

# ------------------------------------------------------------
# extract U,V from scribble and make mask
# ------------------------------------------------------------
yuv       = cv2.cvtColor(scribble_img, cv2.COLOR_BGR2YUV)
scrib_U   = yuv[:,:,1].astype(np.float64)
scrib_V   = yuv[:,:,2].astype(np.float64)
gray_3c   = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
scribble_mask = np.any(scribble_img != gray_3c, axis=2)

# ------------------------------------------------------------
# build linear system
# ------------------------------------------------------------
N = h*w
rows, cols, vals = [], [], []
b_u = np.zeros(N, dtype=np.float64)
b_v = np.zeros(N, dtype=np.float64)
gray_norm = gray_img.astype(np.float64)/255.0

for i in tqdm(range(h), desc="Building system"):
    for j in range(w):
        idx = i*w + j
        if scribble_mask[i,j]:
            rows.append(idx); cols.append(idx); vals.append(1.0)
            b_u[idx] = scrib_U[i,j] - 128
            b_v[idx] = scrib_V[i,j] - 128
        else:
            i0, i1 = max(0, i-d), min(h-1, i+d)
            j0, j1 = max(0, j-d), min(w-1, j+d)
            window = gray_norm[i0:i1+1, j0:j1+1]
            var    = float(np.var(window))
            sigma2 = ALPHA*var + REG_EPS

            # accumulate neighbor weights
            Wsum = 0.0
            nbrs = []
            for ii in range(i0, i1+1):
                for jj in range(j0, j1+1):
                    if ii==i and jj==j: continue
                    nidx = ii*w + jj
                    diff = gray_norm[i,j] - gray_norm[ii,jj]
                    w_ij = np.exp(-(diff*diff)/sigma2)
                    Wsum += w_ij
                    nbrs.append((nidx, w_ij))

            if Wsum > 1e-12:
                for nidx, w_ij in nbrs:
                    rows.append(idx); cols.append(nidx); vals.append(-w_ij/Wsum)
            rows.append(idx); cols.append(idx); vals.append(1.0)

W = sparse.csr_matrix((vals,(rows,cols)), shape=(N,N))

# ------------------------------------------------------------
# solve and reconstruct
# ------------------------------------------------------------
U_p = spsolve(W, b_u)
V_p = spsolve(W, b_v)

Y_chan = gray_img.astype(np.uint8)
U_chan = np.clip(U_p+128, 0,255).reshape(h,w).astype(np.uint8)
V_chan = np.clip(V_p+128, 0,255).reshape(h,w).astype(np.uint8)

out_yuv = cv2.merge([Y_chan, U_chan, V_chan])
out_bgr = cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite(result_path, out_bgr)

print(f"Done → {result_path}")