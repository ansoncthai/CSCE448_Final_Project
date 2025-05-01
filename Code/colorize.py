#!/usr/bin/env python3
# ------------------------------------------------------------
# colorize.py – Levin–Lischinski–Weiss scribble colourisation
# (fixed data-range handling, cleaned duplicates, safer paths)
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
GRAY_DIR     = os.path.join("..", "Gray_Scale")
SCRIBBLE_DIR = os.path.join("..", "Scribble")
RESULT_DIR   = os.path.join("..", "Results")

def pos2id(r, c, W): return r * W + c

def neighbours(r, c, H, W, d=2):
    for rr in range(max(0, r-d), min(H, r+d+1)):
        for cc in range(max(0, c-d), min(W, c+d+1)):
            if rr != r or cc != c:
                yield rr, cc

class Colorizer:
    def __init__(self, gray_filename: str, scribble_filename: str, result_filename: str):
        gray_path     = os.path.join(GRAY_DIR,     gray_filename)
        scribble_path = os.path.join(SCRIBBLE_DIR, scribble_filename)
        self.out_path = os.path.join(RESULT_DIR,   result_filename)

        # ---------- load RGB images (uint8) ----------
        rgb_u8   = cv2.cvtColor(cv2.imread(gray_path),     cv2.COLOR_BGR2RGB)
        scrib_u8 = cv2.cvtColor(cv2.imread(scribble_path), cv2.COLOR_BGR2RGB)

        if rgb_u8 is None or scrib_u8 is None:
            raise FileNotFoundError("One of the input files could not be read.")
        if rgb_u8.shape != scrib_u8.shape:
            raise ValueError("Gray and scribble images must be the same size.")

        # ---------- convert to float32 in [0,1] first ----------
        rgb_f   = rgb_u8  .astype(np.float32) / 255.0
        scrib_f = scrib_u8.astype(np.float32) / 255.0

        # ---------- then to YUV (U,V centred on 0) ----------
        self.yuv  = cv2.cvtColor(rgb_f,   cv2.COLOR_RGB2YUV)
        self.syuv = cv2.cvtColor(scrib_f, cv2.COLOR_RGB2YUV)

        self.rgb   = rgb_u8   # keep originals for hint detection
        self.scrib = scrib_u8

    # pixel considered a “hint” if scribble differs noticeably
    def _hint(self, r, c):
        return np.any(np.abs(self.rgb[r, c] - self.scrib[r, c]) > 10)

    def colorize(self):
        H, W = self.yuv.shape[:2]
        N    = H * W
        A  = sparse.lil_matrix((N, N), dtype=np.float32)
        bu = np.full(N, REG_EPS * 0.5, dtype=np.float32)
        bv = bu.copy()

        for r in tqdm(range(H), desc="assembling"):
            for c in range(W):
                idx = pos2id(r, c, W)
                Y   = self.yuv[r, c, 0]

                if self._hint(r, c):                 # scribble pixel
                    A[idx, idx] = 1.0 + REG_EPS
                    bu[idx] += self.syuv[r, c, 1]
                    bv[idx] += self.syuv[r, c, 2]
                    continue

                nbr = list(neighbours(r, c, H, W, d=2))
                Ys  = np.array([self.yuv[rr, cc, 0] for rr, cc in nbr])
                ids = np.array([pos2id(rr, cc, W)   for rr, cc in nbr])

                σ = np.std(Ys)
                w = np.ones_like(Ys) if σ < 1e-6 else np.exp(-((Ys - Y) ** 2) / (2 * σ ** 2))
                w /= w.sum()

                A[idx, idx] = 1.0 + REG_EPS
                A[idx, ids] -= w

        A = A.tocsc()
        U = spsolve(A, bu).reshape(H, W)
        V = spsolve(A, bv).reshape(H, W)

        out_yuv = np.dstack((self.yuv[..., 0], U, V))
        rgb_f   = cv2.cvtColor(out_yuv.astype(np.float32), cv2.COLOR_YUV2RGB)
        return (np.clip(rgb_f, 0, 1) * 255).astype(np.uint8)

# ------------------- CLI wrapper ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python colorize.py <gray_filename> <scribble_filename> <result_filename>")
        sys.exit(1)

    colorizer = Colorizer(sys.argv[1], sys.argv[2], sys.argv[3])
    result    = colorizer.colorize()

    os.makedirs(RESULT_DIR, exist_ok=True)
    cv2.imwrite(colorizer.out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print("✓ Saved to", colorizer.out_path)