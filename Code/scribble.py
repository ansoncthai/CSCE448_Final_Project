#!/usr/bin/env python3
# ---------------------------------------------------------------
# Scribble-based Image Colorization
#   Implementation of Levin, Lischinski & Weiss (SIGGRAPH 2004)
#   Clean version – ARM-64 macOS friendly
# ---------------------------------------------------------------

import os
import cv2 as cv
import numpy as np
from PIL import Image
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ───────────────────────── helper: local stats ─────────────────────────
def get_window_stats(Y: np.ndarray, r: int = 1):
    """Return per-pixel local mean and variance (ε-regularised)."""
    k = 2 * r + 1
    kernel = np.full((k, k), 1.0 / (k * k), dtype=np.float32)

    mean = cv.filter2D(Y.astype(np.float32), -1, kernel, borderType=cv.BORDER_REFLECT)
    var  = cv.filter2D((Y - mean) ** 2, -1, kernel, borderType=cv.BORDER_REFLECT)
    return mean, var + 1e-6  # avoid zero variance


# ─────────────────────── Laplacian construction ────────────────────────
def build_laplacian(Y: np.ndarray, r: int = 1):
    """
    Correlation-based affinity Laplacian from Levin et al.
    Returns an n×n sparse CSR matrix (n = h·w).
    """
    h, w = Y.shape
    n = h * w
    win_sz = (2 * r + 1) ** 2

    mean, var = get_window_stats(Y, r)

    # Over-allocate   (+1 for diagonal per pixel)
    rows = np.empty(n * (win_sz + 1), dtype=np.int32)
    cols = np.empty_like(rows)
    data = np.empty_like(rows, dtype=np.float32)

    idx = 0
    for y in range(h):
        for x in range(w):
            i = y * w + x
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)

            # indices and intensities in the window
            win_y = np.arange(y0, y1)[:, None]
            win_x = np.arange(x0, x1)[None, :]
            win_idx = (win_y * w + win_x).ravel()

            Iy = Y[win_y, win_x]
            μ  = mean[win_y, win_x]
            σ2 = var[y, x]

            # correlation-based weights (eq. 3 in the paper)
            w_rs = 1.0 + ((Iy - μ) * (Y[y, x] - mean[y, x])) / σ2
            w_rs = w_rs.ravel()
            w_rs /= w_rs.sum()

            k = win_idx.size
            rows[idx : idx + k] = i
            cols[idx : idx + k] = win_idx
            data[idx : idx + k] = -w_rs

            # diagonal
            rows[idx + k] = cols[idx + k] = i
            data[idx + k] = 1.0

            idx += k + 1

    return sp.csr_matrix((data[:idx], (rows[:idx], cols[:idx])), shape=(n, n))


# ─────────────────────────── core solver ───────────────────────────────
def solve_colorization(gray_y, scribble_uv, scribble_mask, r=1, λ=100):
    """
    gray_y        : Y channel (H×W)
    scribble_uv   : (H×W×2) chroma from user image
    scribble_mask : binary mask -- 1 where user drew color
    λ             : large weight on constraints (default 100)
    """
    h, w = gray_y.shape
    n = h * w
    L = build_laplacian(gray_y, r)

    # constraint diagonal
    D_c = sp.diags(scribble_mask.ravel() * λ, format='csr')
    A = L + D_c          # final system  (Lx = 0,  Dc adds hard constraints)
    b = (scribble_mask[..., None] * λ * scribble_uv).reshape(-1, 2)

    out_uv = np.empty_like(scribble_uv, dtype=np.uint8)
    for ch in range(2):
        x = spla.spsolve(A, b[:, ch])
        out_uv[..., ch] = np.clip(x.reshape(h, w), 0, 255).astype(np.uint8)

    return out_uv


# ────────────────────── high-level colorizer ───────────────────────────
def colorize(gray_img: np.ndarray, color_img: np.ndarray, mask: np.ndarray):
    """
    gray_img : H×W  uint8  (grayscale)
    color_img: H×W×3 uint8 (RGB scribble image)
    mask     : H×W  {0,1}
    """
    gray_yuv   = cv.cvtColor(cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2YUV)
    color_yuv  = cv.cvtColor(color_img, cv.COLOR_RGB2YUV)

    uv = solve_colorization(
        gray_y   = gray_yuv[..., 0].astype(np.float32),
        scribble_uv = color_yuv[..., 1:3].astype(np.float32),
        scribble_mask = mask.astype(np.float32)
    )

    yuv_out = np.dstack([gray_yuv[..., 0], uv]).astype(np.uint8)
    return cv.cvtColor(yuv_out, cv.COLOR_YUV2RGB)


# ──────────────────────── I/O wrapper ──────────────────────────────────
def process_scribble_file(inp: str, out: str):
    img = cv.imread(inp, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(inp)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # naive scribble mask: pixels whose color deviates >25 from gray
    diff  = np.abs(img.astype(np.int16) - cv.cvtColor(gray, cv.COLOR_GRAY2RGB).astype(np.int16))
    mask  = (diff.max(-1) > 25).astype(np.uint8)
    mask  = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    print("Running color propagation …")
    result = colorize(gray, img, mask)

    Image.fromarray(result).save(out, quality=95)
    print("Saved →", out)


# ───────────────────────────── main ────────────────────────────────────
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    inp  = os.path.join(base, "..", "Images",  "scribble_3.jpg")
    out  = os.path.join(base, "..", "Results", "result_3.jpg")

    try:
        process_scribble_file(inp, out)
    except Exception as e:
        print("✖ Error:", e)


if __name__ == "__main__":
    main()