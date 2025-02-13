#!/usr/bin/env python3
import pint
u = pint.UnitRegistry()

# computing memory requirements for a naive GPU raytracing alg


# npix1 = 256
# npix2 = npix1
# science pixels
npix1 = 30
npix2 = 200
nchan = 2
nrad = 60
nele = 19
nazi = 37
num_obs = 50
ndays = num_obs
num_iter = 500

density = nrad * nazi * nele * 8 / 1e9
densities = ndays * density
# system matrix size
vg = nchan * npix1 * npix2 * density
#                                 intersections         indices   intersection_length
#                                       |                   |       |
vgc = nchan * npix1 * npix2 * (2 * nrad + nele + nazi/2) * (3 * 1 + 4) / 1e9
# * u.bytes
vgs = vg * num_obs
vgcs = vgc * num_obs

# print('Single VG:', vg.to('GB'))
# print('All VGs:', vgs.to('GB'))
print('Single density:', density * 1000, 'MB')
print('All densities:', densities, 'GB')
print('\nSingle VG:', vg, 'GB')
print('All VGs:', vgs, 'GB')
print('\nSingle VG Coordinates:', vgc, 'GB')
print('All VGs Coordinates:', vgcs, 'GB')

# transfer times
transfer_speed = 32 # GB/s
gpu_mem = 100 # GB
# * u.GB / u.s
print('\nSingle VG transfer time:', vg * num_iter / transfer_speed, 's')
print('All VGs transfer time:', vgs * num_iter / transfer_speed, 's')
print('\nSingle VG Coordinates transfer time:', vgc * num_iter / transfer_speed, 's')
print('All VGs coordinates transfer time:', vgcs * num_iter / transfer_speed, 's')