#!/usr/bin/env python3

import numpy as np
from astropy.coordinates import get_body, EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy import wcs
import astropy.units as u

solar_system_ephemeris.set('de432s')

"""
 Ⓔ    Ⓜ
     🮣🮠
    🮣🮠
   🮣🮠
  🮣🮠
 Ⓢ
"""

from glide.common_components.view_geometry import SpaceCraft, CameraWFI, CameraNFI
sc = SpaceCraft('2025-09-30')
sc.add_sensor(CameraWFI())
sc.pointAtEarth()

# t = sc.date
# t = Time('2025-06-01')
# spacecraft position (near L1)
# 2026-06-01 ECI/GCRS from ephemeris file
# sc = SkyCoord(
#     np.array([[154661.172, 1227228.125, 389377.875]]).T,
#     unit='km', frame='gcrs', representation_type='cartesian',
#     obstime=t
# )

# body position
# body = get_body(
#     'earth',
#     time=t,
#     location=EarthLocation.from_geocentric(0, 0, 0, 'km')
# )
# body radius
# body_r = 1800 * u.km

# vector from spacecraft to body
# sc2body = (body.cartesian - sc.position).get_xyz()
# sc2body = (body.cartesian.get_xyz()[:, None] - sc.position * u.km).flatten()
# sc2body /= np.linalg.norm(sc2body)

# fp0, xy = sc.sensor.createrays(sc.sensor.get_pixels())
# fp0 = sc.transformSpaceCrafttoECI(fp0)
# xy = sc.transformSpaceCrafttoECI(xy)
# los = (fp0 - xy).reshape((3, 512, 512))
# los /= np.linalg.norm(los, axis=0)

# los_dot = np.einsum('i,jkl->kl', sc2body, los)
# los_dot = np.dot(sc2body, los.reshape((3, 512*512))).reshape((512, 512))

sc = SpaceCraft('2025-06-01')
sc.add_sensor(CameraWFI())
sc.updatePositionFromFile(sc.date)
sc.pointAtEarth()
from astropy.time import Time
from datetime import timedelta
Δt = timedelta(days=1)
times = Time('2025-09-30') + np.linspace(-100, 100, 300) * Δt

masks = []
for t in times:
    sc.updatePositionFromFile(t)
    sc.pointAtEarth()
    mask_earth = sc.body_mask('earth')
    mask_moon = sc.body_mask('moon')
    masks.append(np.logical_or(mask_earth, mask_moon))
masks = np.array(masks)

from glide.science.plotting import show_gif
import matplotlib.pyplot as plt
show_gif(masks.astype(float))
plt.show()

"""
- operations in SkyCoord - subtraction?
- use of get_xyz necessary?
- why specify observation location when `body` coordinates are GCRS?

# projection notes
https://archive.stsci.edu/fits/users_guide/node58.html
https://danmoser.github.io/notes/gai_fits-imgs.html#coordinates-and-projections
"""