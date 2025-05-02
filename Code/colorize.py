import os, sys
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

GRAY_DIR     = os.path.join("..", "Gray_Scale")
SCRIBBLE_DIR = os.path.join("..", "Scribble")
RESULT_DIR   = os.path.join("..", "Results")


d = 1          
REG_EPS = 1e-6 

if len(sys.argv) != 4:
    print("Usage: python colorize.py <gray_filename> <scribble_filename> <result_filename>")
    sys.exit(1)
gray_name = sys.argv[1]
scribble_name = sys.argv[2]
result_name = sys.argv[3]

gray_path = os.path.join(GRAY_DIR, gray_name)
scribble_path = os.path.join(SCRIBBLE_DIR, scribble_name)
result_path = os.path.join(RESULT_DIR, result_name)
# load images
gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
scribble_img = cv2.imread(scribble_path, cv2.IMREAD_COLOR)
if gray_img is None or scribble_img is None:
    print("could not load input images")
    sys.exit(1)
h, w = gray_img.shape[:2]
if scribble_img.shape[0] != h or scribble_img.shape[1] != w:
    print("size does not match")
    sys.exit(1)

# convert scribble image to YUV to extract U and V values for scribbled pixels
scribble_yuv = cv2.cvtColor(scribble_img, cv2.COLOR_BGR2YUV)
scribble_U = scribble_yuv[:, :, 1].astype(np.float64)
scribble_V = scribble_yuv[:, :, 2].astype(np.float64)

# if pixel is not grayscale, has scribble color
gray_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
scribble_mask = np.any(scribble_img != gray_3channel, axis=2)  # boolean mask of scribbled pixels

N = h * w
# arrays for sparse matrix construction
rows, cols, vals = [], [], []
b_u = np.zeros(N, dtype=np.float64)
b_v = np.zeros(N, dtype=np.float64)
gray_norm = gray_img.astype(np.float64) / 255.0

# build the sparse coefficient matrix laplacian
for i in tqdm(range(h), desc="Constructing linear system"):
    for j in range(w):
        idx = i * w + j 
        if scribble_mask[i, j]:
            # scribble pixel add constraint
            rows.append(idx); cols.append(idx); vals.append(1.0)
            # chrominance offset
            b_u[idx] = scribble_U[i, j] - 128.0
            b_v[idx] = scribble_V[i, j] - 128.0
        else:
            # non-scribble pixel: build smoothness constraints relative to neighbors and
            # determine neighbor window buonds
            i0 = max(i - d, 0)
            i1 = min(i + d, h - 1)
            j0 = max(j - d, 0)
            j1 = min(j + d, w - 1)
            # compute local intensity
            window = gray_norm[i0:i1+1, j0:j1+1]
            var = float(np.var(window))
            # add regularization to avoid zeros
            sigma2 = var + REG_EPS
            # compute weights for each neighbor in the window
            weight_sum = 0.0
            neighbor_weights = [] 
            for ii in range(i0, i1+1):
                for jj in range(j0, j1+1):
                    if ii == i and jj == j:
                        continue 
                    neighbor_idx = ii * w + jj
                    # Weight = exp[-(Y_i - Y_j)^2 / sigma^2]
                    diff = gray_norm[i, j] - gray_norm[ii, jj]
                    w_ij = np.exp(-(diff * diff) / sigma2)
                    weight_sum += w_ij
                    neighbor_weights.append((neighbor_idx, w_ij))
            # normalize weights and add to matrix
            if weight_sum > 1e-12: 
                for (nbr_idx, w_ij) in neighbor_weights:
                    rows.append(idx); cols.append(nbr_idx); vals.append(-w_ij / weight_sum)
            rows.append(idx); cols.append(idx); vals.append(1.0)

W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
# solve for U and V channels
U_prime = spsolve(W, b_u)
V_prime = spsolve(W, b_v)

# combine the intensity channel with (U, V) channels
Y_channel = gray_img.astype(np.uint8)  
U_channel = np.clip(U_prime + 128.0, 0, 255).reshape(h, w).astype(np.uint8)
V_channel = np.clip(V_prime + 128.0, 0, 255).reshape(h, w).astype(np.uint8)

# convert back from YUV to BGR
result_yuv = cv2.merge([Y_channel, U_channel, V_channel])
result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite(result_path, result_bgr)