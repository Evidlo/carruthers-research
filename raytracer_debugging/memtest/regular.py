#!/usr/bin/env python3

from common import *

# ----- Regular Raytracing -----

# look up voxel indices for each ray and multiply by intersection length, then sum

check_mem()
result = x[obs, r, e, a]
result *= lens
result = result.sum(axis=-1)
check_mem('Regular raytracer:')

prof_save('regular')