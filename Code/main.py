#!/usr/bin/env python3
# ---------------------------------------------------------------
# Interactive Scribble-based Colorization GUI
#   – Levin et al., SIGGRAPH 2004 implementation –
#
# Author: ChatGPT for Kimberly Chen & Anson Thai
# ---------------------------------------------------------------

import os
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from tkinter import ttk

import cv2 as cv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image, ImageTk


# ────────── core colour-propagation algorithm ────────── #

def build_laplacian(lum: np.ndarray, eps: float = 1e-7, win_rad: int = 1):
    h, w = lum.shape
    n = h * w
    inds = np.arange(n).reshape(h, w)

    rows, cols, data = [], [], []
    win_sz = (2 * win_rad + 1) ** 2

    for y in range(h):
        y0, y1 = max(0, y - win_rad), min(h - 1, y + win_rad)
        for x in range(w):
            x0, x1 = max(0, x - win_rad), min(w - 1, x + win_rad)

            win_inds = inds[y0:y1 + 1, x0:x1 + 1].ravel()
            win_pix  = lum[y0:y1 + 1, x0:x1 + 1].ravel()
            mu, var  = win_pix.mean(), win_pix.var() + eps / win_sz
            coeff    = (1 + (win_pix - mu) * (lum[y, x] - mu) / var) / win_sz

            rows.extend([inds[y, x]] * win_inds.size)
            cols.extend(win_inds)
            data.extend(coeff)

    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    return sp.diags(np.array(L.sum(axis=1)).ravel()) - L


def solve_channel(L, b, alpha, lam=100.0):
    A   = L + lam * sp.diags(alpha.flatten())
    rhs = lam * (alpha.flatten() * b.flatten())
    return spla.spsolve(A.tocsr(), rhs).reshape(b.shape)


def colorize(gray: np.ndarray, scribble: np.ndarray, mask: np.ndarray):
    h, w = gray.shape
    L    = build_laplacian(gray)
    result = np.zeros((h, w, 3), np.float32)
    alpha  = mask.astype(np.float32)

    for ch in range(3):
        result[:, :, ch] = np.clip(
            solve_channel(L, scribble[:, :, ch], alpha), 0, 1
        )

    # one bilateral pass to tidy artefacts
    for ch in range(3):
        result[:, :, ch] = cv.bilateralFilter(result[:, :, ch], 5, 0.1, 2.0)

    return (result * 255).astype(np.uint8)


# ────────── Tkinter GUI wrapper ────────── #

class ColorizeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scribble-Based Image Colorizer")
        
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Set maximum window size to 90% of screen
        self.max_width = int(screen_width * 0.9)
        self.max_height = int(screen_height * 0.9)
        
        # state
        self.gray_np   = None      # grayscale luminance 0-1
        self.display   = None      # PhotoImage for canvas
        self.scribble  = None      # H×W×3 float32 [0-1] in RGB
        self.mask      = None      # H×W uint8 (not bool)
        self.brush_col = (1.0, 0.0, 0.0)   # default red in RGB
        self.brush_rad = tk.IntVar(value=5)
        self.last_x = None
        self.last_y = None
        self.image_number = None
        self.original_height, self.original_width = None, None

        # layout
        self.make_widgets()

    # --- UI construction --- #
    def make_widgets(self):
        # Create main frame with scrollbars
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Add horizontal and vertical scrollbars
        h_scroll = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        toolbar = ttk.Frame(main_frame, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Pick Colour", command=self.pick_colour).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Brush").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Scale(toolbar, from_=1, to=40, orient="horizontal",
                  variable=self.brush_rad).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(toolbar, text="Colorize ▶", command=self.run_colorize).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.LEFT)

        # Create canvas with scrollbar support
        self.canvas = tk.Canvas(main_frame, bg="gray",
                              xscrollcommand=h_scroll.set,
                              yscrollcommand=v_scroll.set)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Configure scrollbars
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

        # bind drawing events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

    # --- file ops --- #
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        
        # Extract image number from filename (image_X.jpg)
        import re
        match = re.search(r'image_(\d+)\.', path)
        if match:
            self.image_number = match.group(1)
        else:
            self.image_number = "1"  # default if pattern not found
            
        img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img_gray is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        # Store original dimensions
        self.original_height, self.original_width = img_gray.shape

        # normalise and init buffers
        self.gray_np  = img_gray.astype(np.float32) / 255.0
        h, w          = img_gray.shape
        self.scribble = np.zeros((h, w, 3), np.float32)
        self.mask     = np.zeros((h, w), np.uint8)

        # Calculate scaling factor if image is larger than max window size
        scale = 1.0
        if w > self.max_width or h > self.max_height:
            scale_w = self.max_width / w
            scale_h = self.max_height / h
            scale = min(scale_w, scale_h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the image
            img_gray = cv.resize(img_gray, (new_w, new_h))
            self.gray_np = img_gray.astype(np.float32) / 255.0
            self.scribble = np.zeros((new_h, new_w, 3), np.float32)
            self.mask = np.zeros((new_h, new_w), np.uint8)

        # Configure canvas scrolling region
        self.canvas.config(scrollregion=(0, 0, w, h))
        
        # show on canvas
        self.refresh_canvas(self.gray_np_to_rgb())

    def save_scribble(self):
        """Save the current view with scribbles overlaid"""
        if self.gray_np is None or not self.mask.any():
            messagebox.showinfo("Info", "No scribbles to save.")
            return

        # Create the overlay image
        gray_rgb = self.gray_np_to_rgb()
        overlay = (self.scribble * 255).astype(np.uint8)  # Already in RGB
        alpha = self.mask.astype(np.float32) / 255.0 * 0.7
        alpha = alpha[..., np.newaxis]
        combined = (gray_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)

        # Resize to original dimensions if needed
        if combined.shape[:2] != (self.original_height, self.original_width):
            combined = cv.resize(combined, (self.original_width, self.original_height))

        # Save the scribble overlay
        scribble_path = f"../Results/scribble_{self.image_number}.jpg"
        Image.fromarray(combined).save(scribble_path, quality=95)

    def save_result(self):
        if not hasattr(self, "result_img"):
            messagebox.showinfo("Info", "No result to save yet.")
            return
            
        # Automatically save to Results folder with correct name
        result_path = f"../Results/result_{self.image_number}.jpg"
        
        # Convert from BGR to RGB and resize if needed
        result_rgb = cv.cvtColor(self.result_img, cv.COLOR_BGR2RGB)
        if result_rgb.shape[:2] != (self.original_height, self.original_width):
            result_rgb = cv.resize(result_rgb, (self.original_width, self.original_height))
        
        # Save with high quality
        Image.fromarray(result_rgb).save(result_path, quality=95)
        
        # Also save the scribble overlay
        self.save_scribble()
        
        messagebox.showinfo("Saved", f"Saved result and scribbles to Results folder")

    # --- drawing tools --- #
    def pick_colour(self):
        color = colorchooser.askcolor()[0]  # Returns RGB tuple
        if color:
            # Store in RGB (no conversion needed)
            self.brush_col = tuple(c / 255.0 for c in color)

    def paint(self, event):
        if self.gray_np is None:
            return
            
        # Get canvas scroll position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        x, y = int(x), int(y)
        if not (0 <= x < self.scribble.shape[1] and 0 <= y < self.scribble.shape[0]):
            return
            
        r = self.brush_rad.get()
        
        # Draw smooth line if we have a previous point
        if self.last_x is not None and self.last_y is not None:
            # Calculate intermediate points for smooth line
            points = self.get_line_points(self.last_x, self.last_y, x, y)
            for px, py in points:
                cv.circle(self.scribble, (px, py), r,
                         self.brush_col, thickness=-1, lineType=cv.LINE_AA)
                cv.circle(self.mask, (px, py), r, 255, thickness=-1)
        else:
            # Single point
            cv.circle(self.scribble, (x, y), r,
                     self.brush_col, thickness=-1, lineType=cv.LINE_AA)
            cv.circle(self.mask, (x, y), r, 255, thickness=-1)

        self.last_x, self.last_y = x, y

        # draw overlay on display copy
        disp = self.gray_np_to_rgb()
        overlay = (self.scribble * 255).astype(np.uint8)  # Already in RGB
        
        # Create alpha mask for blending
        alpha = self.mask.astype(np.float32) / 255.0 * 0.7
        alpha = alpha[..., np.newaxis]  # Add channel dimension
        
        # Blend using alpha compositing
        disp = (disp * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        self.refresh_canvas(disp)

    def get_line_points(self, x1, y1, x2, y2):
        """Get points along a line using Bresenham's algorithm"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x2, y2))
        return points

    def canvas_release(self, event):
        self.last_x = None
        self.last_y = None

    # --- processing --- #
    def run_colorize(self):
        if self.gray_np is None:
            messagebox.showinfo("Info", "Open a grayscale image first.")
            return
        if not self.mask.any():
            messagebox.showinfo("Info", "Paint at least one scribble before colorizing.")
            return

        self.config(cursor="watch")
        self.update_idletasks()
        try:
            # Convert scribble to BGR for colorize function
            scribble_bgr = cv.cvtColor((self.scribble * 255).astype(np.uint8), cv.COLOR_RGB2BGR).astype(np.float32) / 255.0
            result_bgr = colorize(self.gray_np, scribble_bgr, self.mask)
            # Store result in BGR format
            self.result_img = result_bgr
            # Convert to RGB for display
            result_rgb = cv.cvtColor(result_bgr, cv.COLOR_BGR2RGB)
            
            # Save both result and scribbles automatically
            self.save_result()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.config(cursor="")
            return
        self.config(cursor="")

        self.refresh_canvas(result_rgb)

    # --- helpers --- #
    def gray_np_to_rgb(self):
        """Convert grayscale to RGB"""
        rgb = cv.cvtColor((self.gray_np * 255).astype(np.uint8), cv.COLOR_GRAY2RGB)
        return rgb

    def refresh_canvas(self, img_rgb: np.ndarray):
        """Display RGB image on canvas"""
        pil = Image.fromarray(img_rgb)
        self.display = ImageTk.PhotoImage(pil)
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display)
        self.update_idletasks()


# ────────── main ────────── #

if __name__ == "__main__":
    ColorizeApp().mainloop()