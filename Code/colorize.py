import os, sys
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# Directories (preserved from original structure)
GRAY_DIR     = os.path.join("..", "Gray_Scale")
SCRIBBLE_DIR = os.path.join("..", "Scribble")
RESULT_DIR   = os.path.join("..", "Results")

# Parameters (can be adjusted if needed)
d = 1          # neighborhood radius (d=1 gives 3x3 window, d=2 gives 5x5, etc.)
REG_EPS = 1e-6 # small regularization for intensity variance to avoid zero-division

# Parse command-line arguments
if len(sys.argv) != 4:
    print("Usage: python colorize.py <gray_filename> <scribble_filename> <result_filename>")
    sys.exit(1)
gray_name = sys.argv[1]
scribble_name = sys.argv[2]
result_name = sys.argv[3]

# Construct file paths
gray_path = os.path.join(GRAY_DIR, gray_name)
scribble_path = os.path.join(SCRIBBLE_DIR, scribble_name)
result_path = os.path.join(RESULT_DIR, result_name)

# Load images
gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
scribble_img = cv2.imread(scribble_path, cv2.IMREAD_COLOR)
if gray_img is None or scribble_img is None:
    print("Error: Could not load input images. Please check file names and paths.")
    sys.exit(1)
h, w = gray_img.shape[:2]
# Ensure scribble image dimensions match grayscale image
if scribble_img.shape[0] != h or scribble_img.shape[1] != w:
    print("Error: Scribble image size does not match grayscale image size.")
    sys.exit(1)

# Convert scribble image to YUV to extract U and V values for scribbled pixels
scribble_yuv = cv2.cvtColor(scribble_img, cv2.COLOR_BGR2YUV)
scribble_U = scribble_yuv[:, :, 1].astype(np.float64)
scribble_V = scribble_yuv[:, :, 2].astype(np.float64)

# Identify scribble pixels by comparing scribble image to grayscale image
# (If a pixel's BGR differs from the grayscale value, it has color scribble.)
gray_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
scribble_mask = np.any(scribble_img != gray_3channel, axis=2)  # boolean mask of scribbled pixels

# Prepare the linear system: size N x N (N = total number of pixels)
N = h * w
# Arrays for sparse matrix construction
rows, cols, vals = [], [], []
# Right-hand side vectors for U and V (initialized to 0, will set scribble constraints)
b_u = np.zeros(N, dtype=np.float64)
b_v = np.zeros(N, dtype=np.float64)

# Normalize grayscale intensities to [0,1] range for weight computation
gray_norm = gray_img.astype(np.float64) / 255.0

# Build the sparse coefficient matrix (Laplacian matrix with constraints)
for i in tqdm(range(h), desc="Constructing linear system"):
    for j in range(w):
        idx = i * w + j  # linear index of pixel (i,j)
        if scribble_mask[i, j]:
            # Scribble pixel: add constraint U(idx)=scribble_U, V(idx)=scribble_V
            rows.append(idx); cols.append(idx); vals.append(1.0)
            # Subtract 128 from U,V to use zero as no-color baseline (chrominance offset)
            b_u[idx] = scribble_U[i, j] - 128.0
            b_v[idx] = scribble_V[i, j] - 128.0
        else:
            # Non-scribble pixel: build smoothness constraints relative to neighbors
            # Determine neighbor window bounds (clamped to image borders)
            i0 = max(i - d, 0)
            i1 = min(i + d, h - 1)
            j0 = max(j - d, 0)
            j1 = min(j + d, w - 1)
            # Compute local intensity statistics
            window = gray_norm[i0:i1+1, j0:j1+1]
            var = float(np.var(window))
            # Add regularization to variance to avoid zero (or extremely low) var
            sigma2 = var + REG_EPS
            # Compute weights for each neighbor in the window
            weight_sum = 0.0
            neighbor_weights = []  # will store (neighbor_idx, weight) for normalization
            for ii in range(i0, i1+1):
                for jj in range(j0, j1+1):
                    if ii == i and jj == j:
                        continue  # skip the center pixel itself
                    neighbor_idx = ii * w + jj
                    # Weight = exp[-(Y_i - Y_j)^2 / sigma^2]
                    diff = gray_norm[i, j] - gray_norm[ii, jj]
                    w_ij = np.exp(-(diff * diff) / sigma2)
                    weight_sum += w_ij
                    neighbor_weights.append((neighbor_idx, w_ij))
            # Normalize weights and add entries to matrix
            if weight_sum > 1e-12:  # if there is any weight
                for (nbr_idx, w_ij) in neighbor_weights:
                    rows.append(idx); cols.append(nbr_idx); vals.append(-w_ij / weight_sum)
            # For stability, if weight_sum is 0 (no variation at all), we simply don't add off-diagonals.
            # (This would mean the entire window is constant intensity; color will propagate uniformly.)
            # Finally, add the diagonal entry
            rows.append(idx); cols.append(idx); vals.append(1.0)

# Create the sparse matrix (in CSR format for solving)
W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

# Solve the sparse linear system for U and V channels
# (We solve for U' and V' which are U,V minus 128 baseline)
U_prime = spsolve(W, b_u)
V_prime = spsolve(W, b_v)

# Combine the Y (intensity) channel with solved chrominance (U, V) channels
Y_channel = gray_img.astype(np.uint8)  # original grayscale intensities [0,255]
U_channel = np.clip(U_prime + 128.0, 0, 255).reshape(h, w).astype(np.uint8)
V_channel = np.clip(V_prime + 128.0, 0, 255).reshape(h, w).astype(np.uint8)

# Convert the result from YUV back to BGR for output
result_yuv = cv2.merge([Y_channel, U_channel, V_channel])
result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite(result_path, result_bgr)