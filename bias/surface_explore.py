#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go

img = np.load('images/oob_nfi_l0.npy')[100:512]
dark = np.load('images/dark_nfi_l0.npy')[100:512]

x = dark
s = np.sum(img, axis=1)

# affine transform the dark
a = .997 # CLAUDE slider with configurable range (default [0, 2] 500 steps)
b = 0 # CLAUDE slider with configurable range (default [-10, 10] 500 steps)
dark = dark * .997 + 0


y = img - dark * .997

selected_cols = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]

fig = go.Figure()
for col in selected_cols:
    fig.add_trace(go.Scatter3d(
        x=x[:, col], y=s, z=y[:, col], mode='markers',
        marker=dict(size=2),
        name=f'col {col}'
    ))

fig.update_layout(scene=dict(xaxis_title='Dark', yaxis_title='Sum', zaxis_title='Image'), title='Surface Exploration')
fig.write_html('/www/surface_explore.html', auto_open=False)
