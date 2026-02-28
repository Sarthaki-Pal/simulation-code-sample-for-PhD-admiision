# This code generates an all-sky map, and generates visibilties for multiple pointings which are listed in a file
# The file containing the paths of the fits are passed in argv[2]
# changed on 27.12.23 Asif & Shouvik

import numpy as np
import healpy as hp
import builtins as blt
import time
from sys import argv
import os
from astropy.table import Table
from astropy.io import fits

import params_sim as psim
import params_MWA as MWA
import image_gen as ig

os.nice(5)  # method used to increment the process’s niceness by specified value
start = time.time()  # for time check

psim.params()  # simulation parameters
MWA.params("MWA")  # MWA parameters

if (Amp == 1) and (beta == 0):
    uaps = True
else:
    uaps = False

if (len(argv) == 4) and (not uaps):
    simulation_params_file = argv[3]
    simulation_params = {'Nrea':Nrea, 'NSIDE':NSIDE, 'amp':Amp, 'beta':beta}
    with open(simulation_params_file, 'w') as f:
        for key, value in simulation_params.items():
            f.write(f'{key}={value}\n')

Npix = hp.nside2npix(NSIDE)  # Number of pixels given N_side
res = hp.nside2resol(NSIDE, arcmin=True)  # resolution given N_side
lmax = NSIDE * 3 - 1  # maximum number of \ell multipoles used # what is this ?
print("Nside=", NSIDE, "resolution=", res, "min", Npix, "pixels")

if uaps:
    print('UAPS simulation')
else:
    print(f'APS simulation for Amp={Amp} and beta={beta}')

# GRF sky map generation, use -1 for random seed.
map_grf = np.zeros((Nrea, Npix))

for ii in range(Nrea):
    iseed = 10*ii+1 # 0, 11, ... etc. iseed must change with realizations
    map_grf[ii] = ig.GRF_hpmap_gen(nside=NSIDE, l_max=lmax, nu=nu_c, func=ig.APS_func, iseed=iseed)  # -1 for random
print(f"{Nrea} Sky maps generated.")

in_fits = str(argv[1])
output_dir = argv[2]

head, tail = os.path.split(in_fits)
fitsName = os.path.splitext(tail)[0]

# working with the input file
hdulist = fits.open(in_fits)
data_tmp = hdulist[0].data  # to save visibilities in data
dataT = Table(data_tmp)  # to convert data from array form to table form, readable

nu_c = hdulist[0].header["CRVAL4"] * 1.0e-6
pol = hdulist[0].header["NAXIS3"]
Nchan = hdulist[0].header["NAXIS4"]
dnu_c = hdulist[0].header["CDELT4"] * 1.0e-6

u = dataT["UU"] * 3.0e8  # *nu_c for baseline unit   #*3.e8 (for meter unit)
v = dataT["VV"] * 3.0e8
w = dataT["WW"] * 3.0e8
bln = np.stack((u, v, w), axis=1)
nbl = len(dataT["UU"])

print(f"RA = {ra_ini}")
print(f"DEC = {dec_mwa}")

vis_all = MWA.visgen_mwa_multi(nside=NSIDE, in_map=map_grf, ra_ptg=ra_ini, dec_ptg=dec_mwa, bl_file=bln, nu=nu_c, ang_range=90.0)

for i in range(Nrea):
    for j in range(pol):
        hdulist[0].data["DATA"][:, 0, 0, 0, :1, j, 0] = vis_all[i].T.real.astype(np.float32)[:, np.newaxis]
        hdulist[0].data["DATA"][:, 0, 0, 0, :1, j, 1] = vis_all[i].T.imag.astype(np.float32)[:, np.newaxis]

    if uaps:
        op_filename = f"{output_dir}/Nrea_{Nrea}_UAPS_{i+1}.fits"
    else:
        op_filename = f"{output_dir}/Nrea_{Nrea}_APS_{i+1}.fits"
    hdulist.writeto(op_filename, overwrite=True)

hdulist.close()

# time check
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("\nTotal time taken -", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
