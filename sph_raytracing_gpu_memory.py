#!/usr/bin/env python3
import pint
u = pint.UnitRegistry()

# computing memory requirements for a naive GPU raytracing alg


nchan = 2
ndays = 365
npix = 512
nrad = 60
nele = 19
nazi = 37
num_obs = 40
num_iter = 500


density = nrad * nazi * nele * 8 / 1e9
densities = ndays * density
# system matrix size
vg = nchan * npix * npix * density
vgc = nchan * npix * npix * ((2 * nrad * 3) * 1 + 8) / 1e9
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