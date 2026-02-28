import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import healpy as hp
import sys
import builtins as blt
import multiprocessing as mul
from functools import partial

import params_sim as psim

psim.params()  # simulation parameters


def APS_func(l, nu):
    """Cl function

    Args:
        l : Multipole
        nu : Frequency

    Returns:
        Returns APS value for each l. This returns a double precision number.
    """
    amplitude = Amp
    pw = beta
    kk = amplitude * l**pw
    return kk


def GRF_hpmap_gen(nside, l_max, nu, func, iseed=-1):
    """This function generates all-sky maps of a Gaussian Random field.

    Args:
        nside : Healpix parameter
        l_max : Maximum l values for the angular power spectrum (APS)
        nu : Frequency
        func : Functional form of APS.
        iseed : Input seed for random field generation. Defaults to -1.

    Returns:
        Returns a HealPix map (i.e.  a numpy array)
    """
    dim = l_max
    kk = np.zeros(dim)

    for ii in range(1, dim):
        kk[ii] = func(ii, nu)
    if iseed != -1:
        np.random.seed(iseed)

    map_grf = hp.synfast(kk, nside=nside, lmax=dim)
    # hp.write_map("test.fits",map, nest=False,coord = 'C')
    return map_grf


def pbeam_hpmap_gen(nside, ra_ptg, dec_ptg, nnu, func):
    """This function generates the primary beam pattern centered at specified RA and DEC. It assumes the telescope is fixed on the ground and points vertically upward. This implies that DEC corresponds to the latitude of the telescope and RA is at zenith.

    Args:
        nside : Healpix parameter
        ra_ptg : pointing RA
        dec_ptg : pointing DEC
        nnu : Frequency
        func : Beam function, returns a double precision number.

    Returns:
        Returns a HealPix map of beam pattern (i.e. a numpy array)
    """
    Npix = hp.nside2npix(nside)
    pix_num = list(range(0, Npix))
    in_map = np.ones(Npix)
    beam_cal_red = partial(beam_cal, func=func, nside=nside, ra0=ra_ptg, dec0=dec_ptg, nu=nnu)
    pool = mul.Pool()
    out_map = np.array(pool.map(beam_cal_red, pix_num))
    pool.close()
    return out_map


def beam(func, ap, dp, a0, d0, nnu):
    """This function makes the beam. This function first calculates the basis vectors (e1, e2 & e3) in a local tangent plane at the phase center, 
    secondly calculates the unit vector along the phase centre and thirdly calculates the difference vector (dn) between the pixel location in the 
    sky and the phase centre. Lastly it calculates the projection of thedifference vector (dn) along e1 and e2 which are denoted by dne1 & dne2.

    Args:
        func : This is the Primary beam of the MWA
        ap : pixel RA in the sky map
        dp : pixel Dec in the sky map
        a0 : RA of the phase centre
        d0 : Phase center dec of the observation w.r.t. dec_mwa
        nnu : Central frequency in MHz

    Returns:
        Returns the primary beam of the MWA along the two differencevectors (dne1 & dne2).
    """
    e1 = np.array([-np.sin(a0), np.cos(a0), 0])
    e2 = np.array([-np.cos(a0) * np.sin(d0), -np.sin(d0) * np.sin(a0), np.cos(d0)])
    e3 = np.array([np.cos(a0) * np.cos(d0), np.sin(a0) * np.cos(d0), np.sin(d0)])

    n = np.array([np.cos(ap) * np.cos(dp), np.sin(ap) * np.cos(dp), np.sin(dp)])
    p = e3

    dn = n - p

    dne1 = np.inner(dn, e1)
    dne2 = np.inner(dn, e2)
    if np.inner(n, p) >= 0:
        return func(nnu, dne1, dne2)
    else:
        return 0.0


def check_dis(lon, ra_ptg):  # check for the upper hemisphere
    flag = 0
    sep = 0.0
    dd = abs(lon - ra_ptg)
    if dd <= 90.0:
        flag = 1
        sep = dd
    elif dd > 180 and (360 - dd) <= 90:
        flag = 1
        sep = 360 - dd
    return flag


def beam_cal(ipix, func, nside, ra0, dec0, nu):
    """This function defines the pixel RA & Dec, phase centre RA & Dec (w.r.t MWA Declination)

    Args:
        ipix :  the sky map pixels
        func : This is the Primary beam of the MWA
        nside : Healpix parameter
        ra0 : RA of the phase centre
        dec0 : Dec of the phase centre
        nu : Central frequency in MHz

    Returns:
        Calls the beam()function and returns it.
    """
    # lon, lat = hp.pix2ang(nside, ipix, lonlat = True)
    # flag= check_dis(lon, ra0)
    # if (flag ==1):
    RA, dp = hp.pix2ang(nside, ipix, lonlat=True)

    ap = np.deg2rad(RA)  # pixel RA
    dp = np.deg2rad(dp)  # pixel dec
    a_0 = np.deg2rad(ra0)  # RA of the phase center
    d_0 = np.deg2rad(dec0)  # Phase center dec of the observation w.r.t. dec_mwa

    o_map = beam(func, ap, dp, a_0, d_0, nu)
    # else :
    # 	o_map = 0.0 #hp.UNSEEN
    return o_map


# def beam_cal(ipix, func, nside, ra0, dec0, nu):
# 	lon, lat = hp.pix2ang(nside, ipix, lonlat = True)
# 	flag= check_dis(lon, ra0)
# 	if (flag ==1):
# 		RA, dp = hp.pix2ang(nside, ipix, lonlat = True)

# 		ap = np.deg2rad(RA) #pixel RA
# 		dp = np.deg2rad(dp)	#pixel dec
# 		a_0 = np.deg2rad(ra0)	#RA of the phase center
# 		d_0 = np.deg2rad(dec0) #Phase center dec of the observation w.r.t. dec_mwa

# 		o_map = beam(func, ap, dp, a_0, d_0, nu)
# 	else :
# 		o_map = 0.0 #hp.UNSEEN
# 	return(o_map)


def view_map(in_map, scale="n", view="E"):
    """A function to plot HealPix map.

    Args:
        in_map : HealPix map (i.e. a numpy array)
        scale : 'n' for natural or 'dB' for decibel. Defaults to "n".
        view : _description_. Defaults to "E".
    """
    if scale == "dB":
        if view == "E":
            hp.mollview(10.0 * np.log10(in_map), coord="C", rot=(0, 0, 0), nest=False, title="map")
        elif view == "G":
            hp.orthview(10.0 * np.log10(in_map), coord="C", rot=(0, 0, 0), nest=False, title="map")
    elif scale == "n":
        if view == "E":
            hp.mollview(in_map, coord="C", rot=(0, 0, 0), nest=False, title="map")
        elif view == "G":
            hp.gnomview(in_map, coord="C", rot=(0, 0, 0), nest=False, title="map")
    else:
        if view == "E":
            hp.mollview(in_map, coord="C", rot=(0, 0, 0), nest=False, title="map")
        elif view == "G":
            hp.gnomview(in_map, coord="C", rot=(0, 0, 0), nest=False, title="map")
    hp.graticule()
    plt.show()
