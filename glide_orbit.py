#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

from glide.common_components.generate_view_geom import *
from glide.science.forward_sph import sc2vg, ScienceGeom
from glide.science.model_sph import DefaultGrid

from sph_raytracer import Operator

sc = gen_mission(num_obs=50, duration=180)

# %% operator

grid = DefaultGrid()
vg = sum(ScienceGeom(s, (50, 100)) for s in sc)

op = Operator(grid, vg, _compute=False)

op.plot().save('/www/out.gif')