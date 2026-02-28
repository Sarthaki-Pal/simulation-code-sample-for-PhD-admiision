import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import healpy as hp
import sys
import builtins as blt
import multiprocessing as mul
from functools import partial
import multiprocessing.pool
import time
import numexpr as ne

import params_sim as psim

psim.params()  # simulation parameters


def read_baseline(filename):
    data = np.loadtxt(filename)
    mm, nn = data.shape
    # print(mm, nn)
    # Baselines are in meter units (U, V, W)
    return(data)

# def check_dis(lon, ra_ptg):
# 	flag = 0
# 	sep = 0.0
# 	dd = abs(lon - ra_ptg)
# 	if (dd <= 90.0):
# 		flag = 1
# 		sep = dd
# 	elif (dd > 180 and (360 - dd)<=90):
# 		flag = 1
# 		sep = 360 - dd
# 	return(flag, sep)


def dn_comp(ipix, a0, d0):
    RA, dp = hp.pix2ang(NSIDE, ipix, lonlat=True)

    ap = np.deg2rad(RA)  # pixel RA
    dp = np.deg2rad(dp)  # pixel dec
    a0 = np.deg2rad(a0)  # RA of the phase center
    d0 = np.deg2rad(d0)  # Phase center dec of the observation

    e1 = np.array([-np.sin(a0), np.cos(a0), 0])
    e2 = np.array([-np.cos(a0)*np.sin(d0), -np.sin(d0)*np.sin(a0), np.cos(d0)])
    e3 = np.array([np.cos(a0)*np.cos(d0), np.sin(a0)*np.cos(d0), np.sin(d0)])

    n = np.array([np.cos(ap)*np.cos(dp), np.sin(ap)*np.cos(dp), np.sin(dp)])
    p = e3

    dn = n - p

    dne1 = np.inner(dn, e1)
    dne2 = np.inner(dn, e2)
    dne3 = np.inner(dn, e3)

    return(dne1, dne2, dne3)


def vis_pix_sum(ipix, dn, i_map, bl, nu):
    dne1, dne2, dne3 = dn[ipix, :]
    ble1, ble2, ble3 = bl
    vis = i_map[ipix] * np.exp(1j*2.0*np.pi * (dne1*ble1 + dne2*ble2 + dne3*ble3))
    return(vis)


# def vis_comp(nside, in_map, ra_ptg, dec_ptg, bln, nu):

#     Npix = hp.nside2npix(nside)  # number of pixels
#     # a list of pixel numbers for multiprocessing
#     pix_num = list(range(0, Npix))
#     dOmega = hp.nside2resol(nside)**2.0  # resolution in s.rad

#     # ------- make array of dn to be used for vis_cal ----#
#     pool = mul.Pool(processes=Nthreads)
#     dn_red = partial(dn_comp, a0=ra_ptg, d0=dec_ptg)

#     #dn = pool.map(dn_red, pix_num)
#     # Storing \Delta n values for iterative use
#     dn = np.array(pool.map(dn_red, pix_num))
#     pool.close()
#     #--------- done make dn array --------------------#

#     lam = C/nu  # wavelength in m
#     Q_nu = 2.0 * KB / lam**2.0  # in Jy/mK
#     #print('Q_nu=', Q_nu)

#     bln = bln/lam  # baselines in units of wavelength
#     mm, nn = bln.shape  # number of baselines

#     #in_map_list = in_map.tolist()
#     vis = np.zeros((mm), dtype=np.complex64)  # assigning visibility array

#     for ii in range(0, mm):
#         vis_sum = partial(vis_pix_sum, dn=dn, i_map=in_map,
#                           bl=bln[ii, :], nu=nu)
#         # If RAM requirements are higher, we may need to modify the upper line.
#         pool = mul.Pool(processes=Nthreads)  # number of threads to use

#         # Complex visibility for ii^{th} baseline
#         vis[ii] = Q_nu * dOmega * sum(pool.map(vis_sum, pix_num))

#         pool.close()  # free the threads

#         # print(ii, 'th baseline done')	#Just for check

#     return(vis)


def vis_write(filename, vis, ra):
    dim, = ra.shape
    if dim == 1:
        out = np.insert(vis, 0, ra, axis=0)  # first col is the RA values
    else:
        out = np.insert(vis, 0, ra, axis=1)  # first col is the RA values
    np.save(filename, out)
    return(0)


def vis_read(filename):
    vis = np.load(filename)
    nn = vis.ndim
    if (nn == 1):
        ra = abs(vis[0])  # Read the first col for RA
        # Remove RA values only Complex visibility remains
        vis = np.delete(vis, 0, 0)

    else:
        ra = abs(vis[:, 0])  # Read the first col for RA
        # Remove RA values only Complex visibility remains
        vis = np.delete(vis, 0, axis=1)

    return(ra, vis)


def vis_write_raw(filename, vis, bl, ra):
    rec, axis = bl.shape
    dim = ra.size

    if dim == 1:
        out = np.insert(vis, 0, ra, axis=0)  # first col is the RA values
    else:
        out = np.insert(vis, 0, ra, axis=1)  # first col is the RA values
        out = out.flatten()

    out_n = np.array([np.real(out), np.imag(out)])
    out_nn = np.transpose(out_n)
    blf = bl.flatten()

    oarray = np.insert(blf, 0, dim, axis=0)  # number of pointings
    # first value is the number of baselines
    oarray = np.insert(oarray, 0, rec, axis=0)
    oarray = np.append(oarray, out_nn)

    f = open(filename, "wb")
    oarray.tofile(f)
    f.close()
    return(0)


###############################################################################

# def vis_pix_sum_fast(dn, i_map, bl, nu):

#     #dne1, dne2, dne3 = dn[ipix, :]
#     ble1, ble2, ble3 = bl

#     vis = sum(i_map[:] * np.exp(1j*2.0*np.pi *
#               (dn[:, 0]*ble1 + dn[:, 1]*ble2 + dn[:, 2]*ble3)))  # notice

#     return(vis)


def vis_pix_sum_supfast(dn, i_map, bl, nu):
    ble1, ble2, ble3 = bl 
    ppi = np.pi #3.141592653589793
    dn0, dn1, dn2 = dn[:, 0].astype(np.float32), dn[:, 1].astype(np.float32), dn[:, 2].astype(np.float32)
    phase = ne.evaluate('exp(1j*2.0*ppi*(dn0*ble1 + dn1*ble2 + dn2*ble3))')
    phaseT = phase.T.astype(np.complex64)
    wmap = ne.evaluate('i_map * phaseT')
    vis = np.sum(wmap)
    return(vis)


def vis_bl_comp(ibl, nside, dn, i_map, bl, nu):
    Npix = hp.nside2npix(nside)  # number of pixels
    # a list of pixel numbers for multiprocessing
    pix_num = list(range(0, Npix))
    dOmega = hp.nside2resol(nside)**2.0  # resolution in s.rad
    # Complex visibility for ii^{th} baseline
    vis_bl = vis_pix_sum_supfast(dn=dn, i_map=i_map, bl=bl[ibl, :], nu=nu)
    return(vis_bl)


# check for the given angular range (+/- ang_range) in deg
def check_dis_ra(lon, ra_ptg, ang_range=90.0):
    flag = 0
    sep = 0.0
    dd = abs(lon - ra_ptg)
    if (dd <= ang_range):
        flag = 1
        sep = dd
    elif (dd > 180 and (360 - dd) <= ang_range):
        flag = 1
        sep = 360 - dd
    return(flag)


# check for the given angular range (+/- ang_range) in deg
def check_dis_dec(lat, dec_ptg, ang_range=90.0):
    flag = 0
    sep = 0.0
    dd = abs(lat - dec_ptg)
    if (dd <= ang_range):
        flag = 1
        sep = dd
    elif (dd > 90 and (180 - dd) <= ang_range):
        flag = 1
        sep = 180 - dd
    return(flag)


def check_pix(ipix, nside, ra_ptg, dec_ptg, ang_range):
    lon, lat = hp.pix2ang(nside, ipix, lonlat=True)
    flag = check_dis_ra(lon, ra_ptg, ang_range)
    flag_dec = check_dis_dec(lat, dec_ptg, ang_range)
    flag = flag * flag_dec
    return (flag)

"""
def vis_comp_fast(nside, in_map, ra_ptg, dec_ptg, bln, nu, ang_range=90.0):

    Npix = hp.nside2npix(nside)  # number of pixels
    # a list of pixel numbers for multiprocessing
    pix_num = list(range(0, Npix))
    dOmega = hp.nside2resol(nside)**2.0  # resolution in s.rad
    ## only working on the 
    MWA_vec = hp.ang2vec(0, dec_ptg, lonlat=True)
    ipix_mask = hp.query_disc(nside=nside, vec=MWA_vec, radius=np.radians(90))
    # print("range=", ang_range)

    # ------- make array of dn to be used for vis_cal ----#
    pool = mul.Pool(processes=Nthreads)
    dn_red = partial(dn_comp, a0=0, d0=dec_ptg)

    # Storing \Delta n values for iterative use
    dn = np.array(pool.map(dn_red, pix_num))
    pool.close()
    pool.join()
    #--------- done make dn array --------------------#

    lam = C/nu  # wavelength in m
    Q_nu = 2.0 * KB / lam**2.0  # in Jy/mK
    # print('Q_nu=', Q_nu)

    bln = bln/lam  # baselines in units of wavelength
    mm, nn = bln.shape  # number of baselines
    # a list of baseline numbers for multiprocessing
    bl_num = list(range(0, mm))

    #in_map_list = in_map.tolist()
    vis = np.zeros((mm), dtype=np.complex64)  # assigning visibility array

    # Just to check the time elapsed for the baseline loop
    start = time.time()  # for time check

    # for ii in range (0, mm):
    vis_bl = partial(vis_bl_comp, nside=nside, dn=dn[ipix_mask, :], i_map=in_map[ipix_mask], bl=bln, nu=nu)
    # If RAM requirements are higher, we may need to modify the upper line.
    pool1 = mul.Pool(processes=Nthreads1)  # number of threads to use

    # vis[ii] = Q_nu * dOmega * sum(pool.map(vis_sum, pix_num)) #Complex visibility for ii^{th} baseline
    vis = Q_nu * dOmega * np.array(pool1.map(vis_bl, bl_num))
    pool1.close()  # free the threads
    pool1.join()

    # print(ii, 'th baseline done')	#Just for check
    # just to ckeck the time elapsed for the baseline loop
    # for time check
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken in baseline loop")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    return(vis)
"""
