#!/usr/bin/env python3
# Preview real L1C vs simulated rectangular images side by side in domrep sliders.
# Both panels are in phot/s/cm²/sr. No registration yet — just eyeballing the
# difference between measured and modeled images.

import pickle
import base64
from pathlib import Path

import numpy as np
import xarray as xr
import torch as t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm

from glide.science.forward_sph import *
from glide.science.model_sph import *
from domrep import *

from load import load

device = 'cuda'
datapath = Path('/home/alex/carruthers/pseudo_l1c_data/')

# --- date range to preview (small for now) ---
# start = np.datetime64('2026-03-20').astype('datetime64[ns]').astype(float); month = 'march'
# end = np.datetime64('2026-03-22').astype('datetime64[ns]').astype(float)
start = np.datetime64('2026-01-19').astype('datetime64[ns]').astype(float); month = 'january'
end = np.datetime64('2026-01-21').astype('datetime64[ns]').astype(float)

dates = np.linspace(start, end, 100).astype('datetime64[ns]')


# --- truth model + grids for the simulator ---
mask = {'WFI': (3, 25), 'NFI': (3, 25)} # valid grid regions for computing masks
sgrid = DefaultGrid((500, 45, 60), size_r=(3, 25), spacing='log', mask_rs=mask)
rgrid = DefaultGrid((200, 45, 60), size_r=(3, 25), spacing='log', mask_rs=mask)
truth = Zoennchen24Model(grid=sgrid, device=device)()

# white for masked/out-of-FOV pixels instead of LogNorm's salt-pepper
cmap = plt.cm.inferno.copy()
cmap.set_bad('white')


def imfig(im, pm, norm):
    # masked -> nan -> white; clip kept pixels up to vmin so sub-vmin/≤0 noise
    # doesn't render as bad/black speckle under LogNorm
    disp = np.where(pm, np.clip(im, norm.vmin, None), np.nan)
    fig, ax = plt.subplots(figsize=(4, 4))
    ai = ax.imshow(disp, norm=norm, cmap=cmap, origin='lower')
    ax.axis('off')
    fig.colorbar(ai, ax=ax)
    fig.tight_layout(pad=0)
    out = plot(fig, height=320)
    plt.close(fig)
    return out


with document('Pseudo-L1C Preview') as doc:
    with itemgrid(length=2, flow='row'):
        for name in ['wfi.zarr', 'nfi.zarr']:
            ds = load(datapath / name, dates)
            real = ds['im'].values  # (n,w,h) phot/s/cm²/sr

            f = ForwardSph(ds['scraft'].values, sgrid=sgrid, rgrid=rgrid, device=device, _compute=False)

            # fixed colorbar shared by real + sim, from real finite/positive pixels
            finite = real[np.isfinite(real) & (real > 0)]
            norm = LogNorm(*np.percentile(finite, [5, 99]))

            labels = [str(d)[:16] for d in ds['time'].values]

            with tags.div():
                tags.h1(name)
                with caption('Real L1C'):
                    with slider(labels=labels):
                        for pm, im in tqdm(zip(f.proj_mask, real)):
                            imfig(im, pm.cpu().numpy(), norm)


                """
                # Rectangular simulated images (phot/s/cm²/sr) for each spacecraft pose
                sim = f.simulate(truth, disable_noise=True)  # list of (w,h), Rayleighs
                # Rayleighs (omni) -> phot/s/cm²/sr:  R*1e6 -> phot/[4π sr];  /4π -> /sr
                sim =  [np.asarray(s) * 1e6 / (4 * np.pi) for s in sim]
                with caption('Simulated'):
                    with slider(labels=labels):
                        for pm, im in tqdm(zip(f.proj_mask, sim)):
                            imfig(im, pm.cpu().numpy(), norm)
                """

                # caption('Real L1C', slider(*[imfig(im, norm) for im in tqdm(real)], labels=labels))
                # caption('Simulated', slider(*[imfig(im, norm) for im in tqdm(sim)], labels=labels))

out = Path('/www/storm/preview_march.html')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc.render())
print(f'Saved to {out}')
