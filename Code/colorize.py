#!/usr/bin/env python3
# ------------------------------------------------------------
# colorize.py  –  Levin–Lischinski–Weiss scribble colourisation
# Now with ε-regularisation to eliminate UV runaway artefacts.
# Usage:
#   python colorize.py <gray_filename> <scribble_filename> <result_filename>
# ------------------------------------------------------------
import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve

REG_EPS = 1e-3          # neutral-chroma regulariser strength

def pos2id(r, c, W): return r * W + c

def neighbours(r, c, H, W, d=2):
    for rr in range(max(0, r-d), min(H, r+d+1)):
        for cc in range(max(0, c-d), min(W, c+d+1)):
            if rr != r or cc != c:
                yield rr, cc

class Colorizer:
    def __init__(self, gray_filename, scribble_filename, result_filename):
        # Build full paths
        gray_path     = os.path.join("..", "Gray_Scale", gray_filename)
        scribble_path = os.path.join("..", "Scribble", scribble_filename)
        self.out_path = os.path.join("..", "Results", result_filename)

        # Load images
        self.rgb   = cv2.cvtColor(cv2.imread(gray_path), cv2.COLOR_BGR2RGB)
        self.scrib = cv2.cvtColor(cv2.imread(scribble_path), cv2.COLOR_BGR2RGB)

        if self.rgb is None or self.scrib is None:
            raise FileNotFoundError("One of the input files could not be read.")

        if self.rgb.shape != self.scrib.shape:
            raise ValueError("Gray and scribble images must be the same size.")

        # Convert to YUV
        self.yuv  = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YUV) / 255.0
        self.syuv = cv2.cvtColor(self.scrib, cv2.COLOR_RGB2YUV) / 255.0

        if self.rgb is None or self.scrib is None:
            raise FileNotFoundError("Could not read one of the input images.")

        if self.rgb.shape != self.scrib.shape:
            raise ValueError("Gray and scribble images must be the same size.")

        self.yuv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2YUV) / 255.0
        self.syuv = cv2.cvtColor(self.scrib, cv2.COLOR_RGB2YUV) / 255.0

    def _hint(self, r, c):
        return np.any(np.abs(self.rgb[r, c] - self.scrib[r, c]) > 10)

    def colorize(self):
        H, W = self.yuv.shape[:2]; N = H * W
        A = sparse.lil_matrix((N, N), dtype=float)
        bu = np.full(N, REG_EPS * 0.5)   # preload RHS with ε·0.5
        bv = bu.copy()

        for r in tqdm(range(H), desc="assembling"):
            for c in range(W):
                idx = pos2id(r, c, W)
                Y   = self.yuv[r, c, 0]

                if self._hint(r, c):
                    A[idx, idx] = 1.0 + REG_EPS
                    bu[idx] += self.syuv[r, c, 1]
                    bv[idx] += self.syuv[r, c, 2]
                    continue

                nbr = list(neighbours(r, c, H, W, d=2))
                Ys  = np.array([self.yuv[rr, cc, 0] for rr, cc in nbr])
                ids = np.array([pos2id(rr, cc, W)   for rr, cc in nbr])

                σ = np.std(Ys)
                w = np.ones_like(Ys) if σ < 1e-6 else np.exp(-((Ys-Y)**2)/(2*σ**2))
                w /= w.sum()

                A[idx, idx] = 1.0 + REG_EPS
                A[idx, ids] -= w

        A = A.tocsc()
        U = spsolve(A, bu).reshape(H, W)
        V = spsolve(A, bv).reshape(H, W)

        out = np.dstack((self.yuv[...,0], U, V))
        rgb = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32),
                           cv2.COLOR_YUV2RGB)
        return (rgb*255).astype(np.uint8)

# ------------------- CLI wrapper ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python colorize.py <gray_filename> <scribble_filename> <result_filename>")
        sys.exit(1)

    # hand all three filenames to the constructor
    colorizer = Colorizer(sys.argv[1], sys.argv[2], sys.argv[3])
    result    = colorizer.colorize()

    # save using the same <result_filename> the user supplied
    cv2.imwrite(sys.argv[3], cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print("✓ Saved", sys.argv[3])