#!/usr/bin/env python3

from common import *

# ----- Expected Memory Usage -----

ind_memory = num_obs * shape * shape * shape / 1e9
ind_memory += num_obs * num_pix * num_pix * num_vox * 3 * 8 / 1e9
lens_memory = num_obs * num_pix * num_pix * num_vox * 8 / 1e9
raytrace_memory = num_obs * num_pix * num_pix * num_vox * 8 / 1e9
print()
print('Index tensor expected:', ind_memory, 'GB')
print('Length tensor expected:', lens_memory, 'GB')
print('Raytracer expected', raytrace_memory, 'GB')

# check that flat/regular results match
# assert result.sum() == result_squeezed.sum(), "Flat/regular results do not match"