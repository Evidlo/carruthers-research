#!/usr/bin/env python3

import matplotlib.pyplot as plt
import tomosipo as ts

from mas.plotting import slider

from glide.science.forward import glide_orbit
from glide.science.model import zoennchen_model, default_geom
from glide.science.plotting import orbit_svg

geom = default_geom()
x = zoennchen_model(geom)

op = ts.operator(geom, glide_orbit())

orbit_svg(geom, op.projection_geometry, rotate=90, scale=1).save('/srv/www/test.svg')

y = op(x)

plt.imshow(y[:, 0])
plt.show()
