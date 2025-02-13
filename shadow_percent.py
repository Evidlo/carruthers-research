#!/usr/bin/env python3
# percent of spherical volume shadowed by earth
# https://arxiv.org/pdf/2203.17227.pdf

import numpy as np

# distance from spacecraft to Earth (Re)
d = 266
# thick radius
R_e = 3

# spherical volume radius
R = 12.5

phi = np.arcsin(R_e / d)

Z1 = -np.cos(phi)*np.sqrt(R**2 - (d * np.sin(phi))**2) + d * np.sin(phi)**2
Z2 =  np.cos(phi)*np.sqrt(R**2 - (d * np.sin(phi))**2) + d * np.sin(phi)**2

p1 = Z1 * np.tan(phi)
p2 = Z2 * np.tan(phi)

vol = np.pi/3 * (
    (R + Z1 + d)**2 * (2*R - Z1 - d)
    + (Z2 - Z1) * (p1**2 + p1*p2 + p2**2)
    + (R - Z2 - d)**2 * (2*R + Z2 + d)
)

print(vol / ((4/3) * np.pi * R**3))

print(
    (np.pi * R_e**2 * 2 * R) /
    ((4/3) * np.pi * R**3)
)
