#!/usr/bin/env python3
# plot spherical harmonics basis

import torch as t
import matplotlib.pyplot as plt
from pathlib import Path

from sph_raytracer.plotting import preview3d
from sph_raytracer.geometry import SphericalGrid

from glide.science.model_sph import SphHarmModel
from glide.science.plotting import save_gif, color_negative, rescale_max

g = SphericalGrid(shape=(10, 50, 50))
m = SphHarmModel(g, max_l=3, device='cuda')
c = t.zeros(m.coeffs_shape, dtype=t.float64, device=m.device)

# output dir
outdir = Path('/www/harm')

# size of each basis plot in pixels
shape = (64, 64)

# --- Plot each basis function ---

previews = []
for i in range(m.coeffs_shape[0]):
    c[:] = 0
    c[i, :] = 1
    preview = preview3d(color_negative(m(c)), g, shape=shape).numpy()
    previews.append(preview)
    # animated
    save_gif(f:=outdir/f'harm{i:02}.gif', preview)
    # still (use first frame)
    preview = rescale_max(preview, 'frame', maxval=1)
    plt.imsave(f:=outdir/f'harm{i:02}.png', preview[0])
    print('wrote', f)

# --- Plot pyramid of bases ---

# combined pyramid
from PIL import Image
import numpy as np

# organize by l and m
max_l = m.max_l
pyramid = {}
for i in range(m.coeffs_shape[0]):
    l = m.l[i].item()
    order = m.m[i].item()
    # normalize each preview for better saturation
    pyramid[(l, order)] = rescale_max(previews[i], 'frame', maxval=1)

# create pyramid layout
# width is 2*max_l + 1 images, height is max_l + 1 rows
img_h, img_w = shape
pyramid_width = (2 * max_l + 1) * img_w
pyramid_height = (max_l + 1) * img_h
n_frames = previews[0].shape[0]

frames = []
for frame_idx in range(n_frames):
    # create blank frame
    frame = np.ones((pyramid_height, pyramid_width, 3), dtype=np.uint8) * 0

    for l in range(max_l + 1):
        for order in range(-l, l + 1):
            if (l, order) in pyramid:
                # calculate position
                row = l
                # center this row (which has 2*l+1 images)
                col_offset = max_l - l
                col = col_offset + (order + l)

                # place image
                y = row * img_h
                x = col * img_w
                img_data = (pyramid[(l, order)][frame_idx] * 255).astype(np.uint8)
                frame[y:y+img_h, x:x+img_w] = img_data

    frames.append(Image.fromarray(frame))

# save combined gif
frames[0].save(
    f:=outdir/'harm_pyramid.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
print('wrote', f)

# save combined still (first frame)
frames[0].save(f:=outdir/'harm_pyramid.png')
print('wrote', f)